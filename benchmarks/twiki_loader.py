import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from memory.simple_memory import SimpleRAGMemory


def _dummy_twiki_sample() -> Dict[str, Any]:
    return {
        "id": "dummy-2wiki-0",
        "question": "Which actor starred in both Titanic and Inception?",
        "answer": "Leonardo DiCaprio",
        "context": [
            ["Titanic (film)", ["Titanic starred Leonardo DiCaprio and Kate Winslet."]],
            ["Inception", ["Inception is a film featuring Leonardo DiCaprio."]],
        ],
        "supporting_facts": [["Titanic (film)", 0], ["Inception", 0]],
    }


def _normalize_sample_ids(samples: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for s in samples:
        if "id" not in s and "_id" in s:
            s = dict(s)
            s["id"] = s["_id"]
        normalized.append(s)
    return normalized


def load_2wiki(
    split: str = "train",
    fallback_to_dummy: bool = True,
    path: Optional[Union[str, Path]] = None,
):
    if path:
        try:
            p = Path(path)
            data = json.loads(p.read_text())
            return _normalize_sample_ids(data)
        except Exception:
            return [_dummy_twiki_sample()] if fallback_to_dummy else None

    try:
        from datasets import load_dataset
    except Exception:
        return [_dummy_twiki_sample()] if fallback_to_dummy else None

    try:
        ds = load_dataset("xanhho/2WikiMultihopQA")[split]
        return _normalize_sample_ids(ds)
    except Exception:
        return [_dummy_twiki_sample()] if fallback_to_dummy else None


def ingest_twiki_sample(memory: SimpleRAGMemory, sample: Dict[str, Any]) -> List[str]:
    ingested_ids: List[str] = []
    context_entries = sample.get("context", [])
    for ctx_title, sentences in context_entries:
        content = " ".join(sentences)
        metadata = {
            "dataset": "2wiki",
            "sample_id": sample.get("id") or sample.get("_id"),
            "title": ctx_title,
            "type": "paragraph",
            "question": sample.get("question"),
        }
        rec_id = memory.ingest(content, metadata)
        ingested_ids.append(rec_id)
    return ingested_ids
