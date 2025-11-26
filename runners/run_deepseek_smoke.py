"""
Minimal smoke test that wires SimpleRAGMemory to DeepSeek chat via llm_providers.
Requires DEEPSEEK_API_KEY in environment.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_loader import init_from_config


def main() -> None:
    # Inline config to avoid writing secrets to disk.
    config = {
        "embedding": {"provider": "random", "dim": 128},
        "llm": {
            "provider": "deepseek",
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-chat",
            "api_key_env": "DEEPSEEK_API_KEY",
            "temperature": 0.2,
            "max_tokens": 128,
        },
    }
    memory, llm_fn = init_from_config(config)

    # Ingest a couple of tiny facts.
    memory.ingest("Paris is the capital of France.", {"dataset": "toy", "fact": "capital"})
    memory.ingest("The Eiffel Tower is located in Paris.", {"dataset": "toy", "fact": "landmark"})

    question = "What is the capital of France?"
    result = memory.rag_answer(question=question, llm_fn=llm_fn, top_k=2)

    print("Q:", question)
    print("A:", result["answer"])
    print("Used records:", result["used_records"])
    print("Metadata:", result["metadata"])


if __name__ == "__main__":
    main()
