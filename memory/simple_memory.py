import uuid
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError("numpy is required for SimpleRAGMemory") from exc


def _safe_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity, handling zero vectors."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def _get_nested(metadata: Dict[str, Any], path: str) -> Any:
    """Resolve dotted metadata paths like `metadata.sample_id`."""
    current: Any = metadata
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


@dataclass
class MemoryRecord:
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    deleted: bool = False


class SimpleRAGMemory:
    def __init__(self, embed_fn: Callable[[str], np.ndarray]):
        self._records: List[MemoryRecord] = []
        self._embed_fn = embed_fn
        self._logger = logging.getLogger(__name__)

    def ingest(self, content: str, metadata: Dict[str, Any]) -> str:
        rec_id = str(uuid.uuid4())
        embedding = None
        try:
            embedding = self._embed_fn(content)
        except Exception:
            embedding = None
        record = MemoryRecord(id=rec_id, content=content, metadata=dict(metadata), embedding=embedding)
        self._records.append(record)
        self._logger.info("Ingested rec_id=%s sample_id=%s title=%s", rec_id, metadata.get("sample_id"), metadata.get("title"))
        return rec_id

    def _matches_selector(self, record: MemoryRecord, selector: Dict[str, Any]) -> bool:
        for key, expected in selector.items():
            if key == "id":
                if record.id != expected:
                    return False
                continue
            if key.startswith("metadata."):
                meta_key = key.split("metadata.", 1)[1]
                if _get_nested(record.metadata, meta_key) != expected:
                    return False
                continue
            if _get_nested(record.metadata, key) != expected:
                return False
        return True

    def _filtered_records(self, constraints: Optional[Dict[str, Any]]) -> List[MemoryRecord]:
        candidates: List[MemoryRecord] = []
        for rec in self._records:
            if rec.deleted:
                continue
            if constraints and not self._matches_selector(rec, constraints):
                continue
            candidates.append(rec)
        return candidates

    def query(
        self,
        question: str,
        constraints: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> List[MemoryRecord]:
        candidates = self._filtered_records(constraints)
        try:
            q_emb = self._embed_fn(question)
        except Exception:
            q_emb = None

        if q_emb is None:
            self._logger.info("Query fallback (no embedding) constraints=%s top_k=%d returning=%d", constraints, top_k, min(top_k, len(candidates)))
            return candidates[:top_k]

        scored: List[tuple[float, MemoryRecord]] = []
        for rec in candidates:
            if rec.embedding is None:
                continue
            sim = _safe_cosine_similarity(q_emb, rec.embedding)
            scored.append((sim, rec))

        if not scored:
            self._logger.info("Query no embeddings available constraints=%s top_k=%d returning=%d", constraints, top_k, min(top_k, len(candidates)))
            return candidates[:top_k]

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [rec for _, rec in scored[:top_k]]
        self._logger.info("Query constraints=%s top_k=%d returned=%d", constraints, top_k, len(top))
        return top

    def update(
        self,
        selector: Dict[str, Any],
        new_content: Optional[str] = None,
        new_metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        updated = 0
        for rec in self._records:
            if not self._matches_selector(rec, selector):
                continue
            if new_content is not None:
                rec.content = new_content
                try:
                    rec.embedding = self._embed_fn(new_content)
                except Exception:
                    rec.embedding = None
            if new_metadata:
                rec.metadata.update(new_metadata)
            updated += 1
        self._logger.info("Update selector=%s updated=%d", selector, updated)
        return updated

    def forget(self, selector: Dict[str, Any], hard_delete: bool = False) -> int:
        deleted = 0
        remaining: List[MemoryRecord] = []
        for rec in self._records:
            if self._matches_selector(rec, selector):
                deleted += 1
                if hard_delete:
                    continue
                rec.deleted = True
            remaining.append(rec)
        if hard_delete:
            self._records = remaining
        self._logger.info("Forget selector=%s hard_delete=%s count=%d", selector, hard_delete, deleted)
        return deleted

    def stats(self) -> Dict[str, Any]:
        total = len(self._records)
        alive_records = [rec for rec in self._records if not rec.deleted]
        alive = len(alive_records)
        avg_len = float(np.mean([len(rec.content) for rec in alive_records])) if alive_records else 0.0
        return {
            "total_records": total,
            "alive_records": alive,
            "avg_content_len_chars": avg_len,
        }

    def export_artifacts(self, include_embedding: bool = False) -> List[Dict[str, Any]]:
        artifacts: List[Dict[str, Any]] = []
        for rec in self._records:
            item = {
                "id": rec.id,
                "content": rec.content,
                "metadata": rec.metadata,
                "deleted": rec.deleted,
            }
            if include_embedding:
                item["embedding"] = rec.embedding.tolist() if rec.embedding is not None else None
            artifacts.append(item)
        return artifacts

    def rag_answer(
        self,
        question: str,
        llm_fn: Callable[..., str],
        constraints: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        **llm_kwargs: Any,
    ) -> Dict[str, Any]:
        retrieved = self.query(question, constraints=constraints, top_k=top_k)
        context = "\n\n".join(rec.content for rec in retrieved)
        answer = llm_fn(question=question, context=context, **llm_kwargs)
        return {
            "answer": answer,
            "used_records": [rec.id for rec in retrieved],
            "metadata": [rec.metadata for rec in retrieved],
        }
