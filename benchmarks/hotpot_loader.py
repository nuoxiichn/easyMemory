import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from memory.simple_memory import SimpleRAGMemory


def _dummy_hotpot_sample() -> Dict[str, Any]:
    return {
        "id": "dummy-hotpot-0",
        "question": "Who wrote the novel Dune and where was it first published?",
        "answer": "Frank Herbert wrote Dune and it was first published in Analog magazine.",
        "context": [
            ["Dune (novel)", ["Dune is a science fiction novel by Frank Herbert."]],
            ["Analog", ["Analog is the magazine that first published the novel Dune."]],
        ],
        "supporting_facts": [["Dune (novel)", 0], ["Analog", 0]],
    }


def _normalize_sample_ids(samples: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for s in samples:
        if "id" not in s and "_id" in s:
            s = dict(s)
            s["id"] = s["_id"]
        normalized.append(s)
    return normalized


def load_hotpot(
    split: str = "train",
    fallback_to_dummy: bool = True,
    path: Optional[Union[str, Path]] = None,
):
    """
    Load HotpotQA either from a local JSON file (list of dicts) or HF datasets.
    """
    if path:
        try:
            p = Path(path)
            data = json.loads(p.read_text())
            return _normalize_sample_ids(data)
        except Exception:
            return [_dummy_hotpot_sample()] if fallback_to_dummy else None

    try:
        from datasets import load_dataset
    except Exception:
        return [_dummy_hotpot_sample()] if fallback_to_dummy else None

    try:
        ds = load_dataset("hotpotqa/hotpot_qa")[split]
        return _normalize_sample_ids(ds)
    except Exception:
        return [_dummy_hotpot_sample()] if fallback_to_dummy else None


def ingest_hotpot_sample(memory: SimpleRAGMemory, sample: Dict[str, Any]) -> List[str]:
    ingested_ids: List[str] = []
    context_entries = sample.get("context", [])
    for ctx_title, sentences in context_entries:
        content = " ".join(sentences)
        metadata = {
            "dataset": "hotpot",
            "sample_id": sample.get("id") or sample.get("_id"),
            "title": ctx_title,
            "type": "paragraph",
            "question": sample.get("question"),
        }
        rec_id = memory.ingest(content, metadata)
        ingested_ids.append(rec_id)
    return ingested_ids
