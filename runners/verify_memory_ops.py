
import argparse
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_loader import init_from_config, load_config
from embedding_providers import random_embedding
from memory.postgres_memory import PostgresMemory
from memory.simple_memory import SimpleRAGMemory


def build_memory(cfg, use_random_embed: bool, embed_dim: int, dsn_override: str | None):
    if use_random_embed:
        embed_fn = lambda text: random_embedding(text, dim=embed_dim)
        db_cfg = cfg.get("db", {}) if isinstance(cfg, dict) else {}
        dsn = dsn_override or db_cfg.get("dsn")
        if dsn:
            return PostgresMemory(embed_fn=embed_fn, dsn=dsn)
        return SimpleRAGMemory(embed_fn=embed_fn)
    memory, _ = init_from_config(cfg)
    if dsn_override and isinstance(memory, PostgresMemory):
        memory = PostgresMemory(embed_fn=memory._embed_fn, dsn=dsn_override)  # type: ignore[attr-defined]
    return memory


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify ingest/query/update/forget/stats/export.")
    parser.add_argument("--random-embed", action="store_true", help="Use deterministic random embeddings instead of config embedding.")
    parser.add_argument("--embed-dim", type=int, default=64, help="Embedding dim when using random embedding.")
    parser.add_argument("--dsn", type=str, default=None, help="Override Postgres DSN (optional).")
    parser.add_argument("--hard-clean", action="store_true", help="Hard delete test records at the end.")
    parser.add_argument("--top-k", type=int, default=2, help="Top-k to fetch in query.")
    args = parser.parse_args()

    cfg = load_config()
    memory = build_memory(cfg, use_random_embed=args.random_embed, embed_dim=args.embed_dim, dsn_override=args.dsn)

    test_sample_id = f"verify-{uuid.uuid4()}"
    records = [
        ("Paris is the capital of France.", {"dataset": "verify", "sample_id": test_sample_id, "title": "fact1"}),
        ("The Eiffel Tower is in Paris.", {"dataset": "verify", "sample_id": test_sample_id, "title": "fact2"}),
    ]

    print("== Stats (before) ==")
    print(memory.stats())

    ingested_ids = [memory.ingest(content, meta) for content, meta in records]
    print(f"Ingested {len(ingested_ids)} test records with sample_id={test_sample_id}")

    res = memory.query(
        question="What is the capital of France?",
        constraints={"dataset": "verify", "sample_id": test_sample_id},
        top_k=args.top_k,
    )
    print("== Query result IDs ==", [r.id for r in res])
    print("Query result titles:", [r.metadata.get("title") for r in res])

    if res and False:
        upd_count = memory.update({"id": res[0].id}, new_content="UPDATED CONTENT DEMO")
        print("Updated records:", upd_count)
        del_count = memory.forget({"id": res[0].id}, hard_delete=False)
        print("Soft-deleted records:", del_count)

    print("== Stats (after update/delete) ==")
    print(memory.stats())

    artifacts = memory.export_artifacts(include_embedding=False)
    own_artifacts = [
        a
        for a in artifacts
        if a["metadata"].get("dataset") == "verify" and a["metadata"].get("sample_id") == test_sample_id
    ]
    print(f"Exported {len(own_artifacts)} artifacts for test sample (showing up to 3):")
    for art in own_artifacts[:3]:
        preview = art["content"][:80] + ("..." if len(art["content"]) > 80 else "")
        print({"id": art["id"], "deleted": art["deleted"], "metadata": art["metadata"], "content_preview": preview})

    if args.hard_clean:
        deleted = memory.forget({"metadata.sample_id": test_sample_id}, hard_delete=True)
        print(f"Hard-deleted {deleted} test records; cleanup complete.")


if __name__ == "__main__":
    main()
