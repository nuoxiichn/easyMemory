import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from memory.simple_memory import SimpleRAGMemory


def _dummy_mquake_sample() -> Dict[str, Any]:
    return {
        "id": "dummy-mquake-0",
        "knowledge_base": [
            {"id": "fact-0", "text": "Paris is the capital of France."},
            {"id": "fact-1", "text": "The Eiffel Tower is located in Paris."},
        ],
        "edits": [
            {"fact_id": "fact-1", "new_fact": "The Eiffel Tower is located in Paris, France and completed in 1889."}
        ],
        "questions": {
            "pre_edit": "Which city is the Eiffel Tower located in?",
            "post_edit": "When was the Eiffel Tower completed?",
        },
        "answers": {"pre_edit": "Paris", "post_edit": "1889"},
    }


def load_mquake(split: str = "train", fallback_to_dummy: bool = True):
    try:
        from datasets import load_dataset
    except Exception:
        return [_dummy_mquake_sample()] if fallback_to_dummy else None

    try:
        ds = load_dataset("henryzhongsc/MQuAKE-Remastered")[split]
        return ds
    except Exception:
        return [_dummy_mquake_sample()] if fallback_to_dummy else None


def load_mquake_local(path: Union[str, Path]) -> List[Dict[str, Any]]:
    p = Path(path)
    data = json.loads(p.read_text())
    return data


def _fact_id(fact_entry: Any, default_idx: int) -> str:
    if isinstance(fact_entry, dict):
        return str(fact_entry.get("id") or fact_entry.get("fact_id") or f"fact-{default_idx}")
    return f"fact-{default_idx}"


def _fact_text(fact_entry: Any) -> str:
    if isinstance(fact_entry, dict):
        for key in ("fact", "text", "content"):
            if key in fact_entry:
                return str(fact_entry[key])
        if all(k in fact_entry for k in ("question", "answer")):
            return f"{fact_entry['question']} -> {fact_entry['answer']}"
        if "predicate" in fact_entry and "subject" in fact_entry and "object" in fact_entry:
            return f"{fact_entry['subject']} {fact_entry['predicate']} {fact_entry['object']}."
    return str(fact_entry)


def ingest_mquake_sample(memory: SimpleRAGMemory, sample: Dict[str, Any], state: str = "pre_edit") -> List[str]:
    ingested_ids: List[str] = []
    kb = sample.get("knowledge_base") or sample.get("facts") or sample.get("single_hops") or []
    for idx, fact_entry in enumerate(kb):
        content = _fact_text(fact_entry)
        fact_id = _fact_id(fact_entry, idx)
        metadata = {
            "dataset": "mquake",
            "sample_id": sample.get("id") or sample.get("_id") or sample.get("case_id"),
            "fact_id": fact_id,
            "state": state,
        }
        rec_id = memory.ingest(content, metadata)
        ingested_ids.append(rec_id)
    return ingested_ids


def apply_mquake_edits(memory: SimpleRAGMemory, sample: Dict[str, Any]) -> Tuple[int, int]:
    edits = sample.get("edits") or sample.get("new_single_hops") or []
    deleted = 0
    added = 0
    for idx, edit in enumerate(edits):
        fact_id = str(edit.get("fact_id") or edit.get("id") or f"edit-{idx}")
        selector = {
            "metadata.dataset": "mquake",
            "metadata.sample_id": sample.get("id") or sample.get("_id") or sample.get("case_id"),
            "metadata.fact_id": fact_id,
        }
        deleted += memory.forget(selector, hard_delete=True)
        new_content = _fact_text(edit.get("new_fact") or edit.get("fact") or edit.get("text") or edit)
        metadata = {
            "dataset": "mquake",
            "sample_id": sample.get("id") or sample.get("_id") or sample.get("case_id"),
            "fact_id": fact_id,
            "state": "post_edit",
        }
        memory.ingest(new_content, metadata)
        added += 1
    return deleted, added
