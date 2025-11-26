import json
import logging
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import psycopg2
import psycopg2.extras
from psycopg2 import sql

from memory.simple_memory import MemoryRecord, _get_nested


def _match_selector(record: MemoryRecord, selector: Optional[Dict[str, Any]]) -> bool:
    if not selector:
        return True
    for key, val in selector.items():
        if key == "id":
            if record.id != val:
                return False
            continue
        if key.startswith("metadata."):
            meta_key = key.split(".", 1)[1]
            if record.metadata.get(meta_key) != val:
                return False
            continue
        # Fallback: try metadata direct or nested
        if _get_nested(record.metadata, key) != val:
            return False
    return True


class PostgresMemory:
    """
    Simple Postgres-backed memory implementing the same interface as SimpleRAGMemory.
    Embeddings are stored as JSON arrays; metadata as JSONB.
    """

    def __init__(self, embed_fn, dsn: str, table_name: str = "memory_records"):
        self._embed_fn = embed_fn
        self._dsn = dsn
        self._table = table_name
        self._conn = psycopg2.connect(dsn, cursor_factory=psycopg2.extras.RealDictCursor)
        self._conn.autocommit = True
        self._logger = logging.getLogger(__name__)
        self._logger.info("Connecting to PostgresMemory dsn=%s table=%s", dsn, table_name)
        self._ensure_table()

    def _ensure_table(self) -> None:
        table = sql.Identifier(self._table)
        with self._conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {} (
                        id UUID PRIMARY KEY,
                        content TEXT NOT NULL,
                        metadata JSONB,
                        embedding JSONB,
                        deleted BOOLEAN DEFAULT FALSE
                    );
                    """
                ).format(table)
            )

    def ingest(self, content: str, metadata: Dict[str, Any]) -> str:
        rec_id = str(uuid.uuid4())
        try:
            emb = self._embed_fn(content)
            emb_list = emb.tolist() if emb is not None else None
        except Exception:
            emb_list = None
        with self._conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {} (id, content, metadata, embedding, deleted)
                    VALUES (%s, %s, %s::jsonb, %s::jsonb, FALSE)
                    """
                ).format(sql.Identifier(self._table)),
                (rec_id, content, json.dumps(metadata), json.dumps(emb_list) if emb_list is not None else None),
            )
        self._logger.info("Ingested rec_id=%s sample_id=%s title=%s", rec_id, metadata.get("sample_id"), metadata.get("title"))
        return rec_id

    def _fetch_all(self) -> List[MemoryRecord]:
        with self._conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT id, content, metadata, embedding, deleted FROM {}").format(
                    sql.Identifier(self._table)
                )
            )
            rows = cur.fetchall()
        records: List[MemoryRecord] = []
        for row in rows:
            emb = row.get("embedding")
            emb_arr = np.array(emb, dtype=np.float32) if emb is not None else None
            records.append(
                MemoryRecord(
                    id=str(row["id"]),
                    content=row["content"],
                    metadata=row.get("metadata") or {},
                    embedding=emb_arr,
                    deleted=row.get("deleted", False),
                )
            )
        return records

    def query(self, question: str, constraints: Optional[Dict[str, Any]] = None, top_k: int = 5) -> List[MemoryRecord]:
        candidates = [r for r in self._fetch_all() if not r.deleted and _match_selector(r, constraints)]
        if not candidates:
            self._logger.info("Query returned 0 candidates (constraints=%s)", constraints)
            return []
        try:
            q_emb = self._embed_fn(question)
        except Exception:
            q_emb = None

        if q_emb is None:
            return candidates[:top_k]

        q_norm = np.linalg.norm(q_emb)
        scored = []
        for rec in candidates:
            if rec.embedding is None:
                scored.append((0.0, rec))
                continue
            denom = (np.linalg.norm(rec.embedding) * q_norm) or 1e-8
            score = float(np.dot(rec.embedding, q_emb) / denom)
            scored.append((score, rec))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [rec for _, rec in scored[:top_k]]
        if len(top) < top_k and len(top) < len(candidates):
            seen = {r.id for r in top}
            for rec in candidates:
                if rec.id in seen:
                    continue
                top.append(rec)
                if len(top) >= top_k:
                    break
        self._logger.info("Query constraints=%s top_k=%d returned=%d", constraints, top_k, len(top))
        return top

    def update(
        self,
        selector: Dict[str, Any],
        new_content: Optional[str] = None,
        new_metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        records = self._fetch_all()
        matched = [r for r in records if _match_selector(r, selector)]
        count = 0
        for rec in matched:
            content = new_content if new_content is not None else rec.content
            metadata = new_metadata if new_metadata is not None else rec.metadata
            try:
                emb = self._embed_fn(content)
                emb_list = emb.tolist() if emb is not None else None
            except Exception:
                emb_list = None
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE memory_records
                    SET content = %s,
                        metadata = %s::jsonb,
                        embedding = %s::jsonb
                    WHERE id = %s
                    """,
                    (content, json.dumps(metadata), json.dumps(emb_list) if emb_list is not None else None, rec.id),
                )
            count += 1
        self._logger.info("Update selector=%s updated=%d", selector, count)
        return count

    def forget(self, selector: Dict[str, Any], hard_delete: bool = False) -> int:
        records = self._fetch_all()
        matched_ids = [r.id for r in records if _match_selector(r, selector)]
        if not matched_ids:
            return 0
        id_array = matched_ids
        with self._conn.cursor() as cur:
            if hard_delete:
                cur.execute(
                    sql.SQL("DELETE FROM {} WHERE id = ANY(%s::uuid[])").format(
                        sql.Identifier(self._table)
                    ),
                    (id_array,),
                )
            else:
                cur.execute(
                    sql.SQL("UPDATE {} SET deleted = TRUE WHERE id = ANY(%s::uuid[])").format(
                        sql.Identifier(self._table)
                    ),
                    (id_array,),
                )
        self._logger.info("Forget selector=%s hard_delete=%s count=%d", selector, hard_delete, len(matched_ids))
        return len(matched_ids)

    def stats(self) -> Dict[str, Any]:
        with self._conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT
                        COUNT(*) AS total_records,
                        COUNT(*) FILTER (WHERE deleted = FALSE) AS alive_records,
                        COALESCE(AVG(LENGTH(content)), 0) AS avg_content_len_chars
                    FROM {};
                    """
                ).format(sql.Identifier(self._table))
            )
            row = cur.fetchone()
        return {
            "total_records": int(row["total_records"]),
            "alive_records": int(row["alive_records"]),
            "avg_content_len_chars": float(row["avg_content_len_chars"] or 0),
        }

    def export_artifacts(self, include_embedding: bool = False) -> List[Dict[str, Any]]:
        artifacts: List[Dict[str, Any]] = []
        for rec in self._fetch_all():
            entry = {
                "id": rec.id,
                "content": rec.content,
                "metadata": rec.metadata,
                "deleted": rec.deleted,
            }
            if include_embedding:
                entry["embedding"] = rec.embedding.tolist() if rec.embedding is not None else None
            artifacts.append(entry)
        return artifacts

    def rag_answer(self, question: str, llm_fn, constraints: Optional[Dict[str, Any]] = None, top_k: int = 5) -> Dict[str, Any]:
        """
        Convenience helper mirroring SimpleRAGMemory: retrieve then call llm_fn.
        """
        recs = self.query(question, constraints=constraints, top_k=top_k)
        context = "\n\n".join(r.content for r in recs)
        answer = llm_fn(question=question, context=context)
        self._logger.info("RAG answer constraints=%s top_k=%d used=%d", constraints, top_k, len(recs))
        return {
            "answer": answer,
            "used_records": [r.id for r in recs],
            "metadata": [r.metadata for r in recs],
        }

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
