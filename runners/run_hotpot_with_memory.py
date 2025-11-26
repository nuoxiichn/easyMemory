import argparse
import sys
import time
from pathlib import Path
import logging

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.hotpot_loader import ingest_hotpot_sample, load_hotpot
from config_loader import init_from_config, load_config


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run HotpotQA ingest + RAG using configured memory/LLM.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to process.")
    parser.add_argument("--log-interval", type=int, default=5, help="Log progress every N samples (0 to disable).")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k context to retrieve for RAG.")
    args = parser.parse_args()

    config = load_config()
    datasets_cfg = config.get("datasets", {}).get("hotpot", {}) if isinstance(config, dict) else {}
    path = datasets_cfg.get("path")
    split = datasets_cfg.get("split", "validation")

    db_tables = config.get("db", {}).get("tables", {}) if isinstance(config, dict) else {}
    table_name = db_tables.get("hotpot")

    memory, llm_fn = init_from_config(config, table_name=table_name)

    ds = load_hotpot(split=split, path=path, fallback_to_dummy=True)
    if not ds:
        print("Failed to load HotpotQA sample.")
        return

    if args.limit is not None:
        ds = list(ds)[: args.limit]
    else:
        ds = list(ds)

    # Ingest all contexts up front
    for idx, sample in enumerate(ds, start=1):
        ingest_hotpot_sample(memory, sample)
        if args.log_interval and idx % args.log_interval == 0:
            print(f"[progress] ingested sample {idx}/{len(ds)}", flush=True)

    # Prepare report
    reports_dir = ROOT / "reports" / "hotpot"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"report_{int(time.time())}.txt"

    lines = []
    for idx, sample in enumerate(ds, start=1):
        q = sample.get("question", "")
        gold = sample.get("answer", "")
        sample_id = sample.get("id") or sample.get("_id")
        result = memory.rag_answer(
            question=q,
            llm_fn=llm_fn,
            constraints={"dataset": "hotpot", "sample_id": sample_id},
            top_k=args.top_k,
        )
        line = {
            "index": idx,
            "sample_id": sample_id,
            "question": q,
            "gold_answer": gold,
            "pred_answer": result["answer"],
            "used_records": result["used_records"],
            "metadata": result["metadata"],
        }
        lines.append(line)
        if args.log_interval and idx % args.log_interval == 0:
            print(f"[progress] answered sample {idx}/{len(ds)}", flush=True)

    # Write report
    with report_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"Index: {line['index']}\n")
            f.write(f"Sample ID: {line['sample_id']}\n")
            f.write(f"Question: {line['question']}\n")
            f.write(f"Gold: {line['gold_answer']}\n")
            f.write(f"Pred: {line['pred_answer']}\n")
            f.write(f"Used records: {line['used_records']}\n")
            f.write(f"Metadata: {line['metadata']}\n")
            f.write("-" * 40 + "\n")

    print(f"Report saved to: {report_path}")
    print("Final stats:", memory.stats())


if __name__ == "__main__":
    main()
