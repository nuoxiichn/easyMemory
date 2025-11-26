import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.mquake_loader import apply_mquake_edits, ingest_mquake_sample, load_mquake, load_mquake_local
from config_loader import init_from_config, load_config


def _pick_pre(sample: Dict[str, Any]) -> (str, str, str):
    # Use first question in "questions" list and its answer; fallback to top-level answer.
    questions = sample.get("questions") or []
    pre_question = ""
    pre_answer = ""
    if questions and isinstance(questions, list):
        pre_question = questions[0]
        pre_answer = sample.get("answer") or ""
    else:
        pre_question = sample.get("question") or "Pre-edit question missing."
        pre_answer = sample.get("answer") or "N/A"
    return pre_question, pre_answer, sample.get("answer_alias") or []


def _pick_post(sample: Dict[str, Any]) -> (str, str, str):
    # Use new_single_hops[0].question and new_answer if available.
    new_hops = sample.get("new_single_hops") or []
    post_question = ""
    post_answer = sample.get("new_answer") or ""
    if new_hops and isinstance(new_hops, list) and isinstance(new_hops[0], dict):
        post_question = new_hops[0].get("question") or post_question
        post_answer = new_hops[0].get("answer") or post_answer
    if not post_question:
        post_question = sample.get("questions")[0] if sample.get("questions") else "Post-edit question missing."
    if not post_answer:
        post_answer = sample.get("new_answer") or "N/A"
    return post_question, post_answer, sample.get("new_answer_alias") or []


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run MQuAKE ingest/update + RAG using configured memory/LLM.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to process.")
    parser.add_argument("--log-interval", type=int, default=5, help="Log progress every N samples (0 to disable).")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k context to retrieve for RAG.")
    args = parser.parse_args()

    config = load_config()
    datasets_cfg = config.get("datasets", {}).get("mquake", {}) if isinstance(config, dict) else {}
    path = datasets_cfg.get("path")
    split = datasets_cfg.get("split", "validation")

    db_tables = config.get("db", {}).get("tables", {}) if isinstance(config, dict) else {}
    table_name = db_tables.get("mquake")

    memory, llm_fn = init_from_config(config, table_name=table_name)

    if path:
        ds = load_mquake_local(path)
    else:
        ds = load_mquake(split=split, fallback_to_dummy=True)

    if not ds:
        print("Failed to load MQuAKE sample.")
        return

    if args.limit is not None:
        ds = list(ds)[: args.limit]
    else:
        ds = list(ds)

    reports_dir = ROOT / "reports" / "mquake"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"report_{int(time.time())}.txt"

    lines: List[Dict[str, Any]] = []
    for idx, sample in enumerate(ds, start=1):
        sample_id = sample.get("id") or sample.get("_id") or sample.get("case_id") or f"sample-{idx}"
        ingest_mquake_sample(memory, sample, state="pre_edit")

        pre_question, pre_answer, pre_alias = _pick_pre(sample)
        pre_result = memory.rag_answer(
            question=pre_question,
            llm_fn=llm_fn,
            constraints={"dataset": "mquake", "sample_id": sample_id, "state": "pre_edit"},
            top_k=args.top_k,
        )

        deleted, added = apply_mquake_edits(memory, sample)

        post_question, post_answer, post_alias = _pick_post(sample)
        post_result = memory.rag_answer(
            question=post_question,
            llm_fn=llm_fn,
            constraints={"dataset": "mquake", "sample_id": sample_id},
            top_k=args.top_k,
        )

        line = {
            "index": idx,
            "sample_id": sample_id,
            "pre_question": pre_question,
            "pre_gold": pre_answer,
            "pre_pred": pre_result["answer"],
            "pre_alias": pre_alias,
            "pre_used": pre_result["used_records"],
            "pre_metadata": pre_result["metadata"],
            "edits_deleted": deleted,
            "edits_added": added,
            "post_question": post_question,
            "post_gold": post_answer,
            "post_pred": post_result["answer"],
            "post_alias": post_alias,
            "post_used": post_result["used_records"],
            "post_metadata": post_result["metadata"],
        }
        lines.append(line)

        if args.log_interval and idx % args.log_interval == 0:
            print(f"[progress] processed sample {idx}/{len(ds)}", flush=True)

    with report_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"Index: {line['index']}\n")
            f.write(f"Sample ID: {line['sample_id']}\n")
            f.write(f"Pre Q: {line['pre_question']}\n")
            f.write(f"Pre Gold: {line['pre_gold']} | Alias: {line['pre_alias']}\n")
            f.write(f"Pre Pred: {line['pre_pred']}\n")
            f.write(f"Pre Used: {line['pre_used']}\n")
            f.write(f"Pre Metadata: {line['pre_metadata']}\n")
            f.write(f"Edits Deleted: {line['edits_deleted']}, Added: {line['edits_added']}\n")
            f.write(f"Post Q: {line['post_question']}\n")
            f.write(f"Post Gold: {line['post_gold']} | Alias: {line['post_alias']}\n")
            f.write(f"Post Pred: {line['post_pred']}\n")
            f.write(f"Post Used: {line['post_used']}\n")
            f.write(f"Post Metadata: {line['post_metadata']}\n")
            f.write("-" * 40 + "\n")

    print(f"Report saved to: {report_path}")
    print("Final stats:", memory.stats())


if __name__ == "__main__":
    main()
