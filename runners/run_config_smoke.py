"""
Load config/memory_config.yaml, init memory + llm, and run a tiny RAG QA round.
Requires the API keys present in the config or corresponding env vars.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_loader import init_from_config, load_config


def main() -> None:
    config = load_config()
    memory, llm_fn = init_from_config(config)

    # Ingest two toy facts.
    memory.ingest("Paris is the capital of France.", {"dataset": "toy", "fact": "capital"})
    memory.ingest("The Eiffel Tower is located in Paris.", {"dataset": "toy", "fact": "landmark"})

    question = "What is the capital of France?"
    result = memory.rag_answer(question=question, llm_fn=llm_fn, top_k=2)

    print("Q:", question)
    print("A:", result["answer"])
    print("Used records:", result["used_records"])
    print("Metadata:", result["metadata"])
    print("Stats:", memory.stats())


if __name__ == "__main__":
    main()
