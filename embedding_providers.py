import logging
import os
from typing import Any, Callable, Dict, Optional

import requests

import numpy as np

DEFAULT_DIM = 384
logger = logging.getLogger(__name__)


def random_embedding(text: str, dim: int = DEFAULT_DIM, seed: Optional[int] = None) -> np.ndarray:
    """Deterministic per-text random embedding for offline use."""
    base_seed = seed if seed is not None else abs(hash(text)) % (2**32)
    rng = np.random.default_rng(base_seed)
    vec = rng.normal(size=dim).astype(np.float32)
    logger.info("Generated random embedding dim=%d text_len=%d", dim, len(text))
    return vec


def openai_compatible_embedding(
    text: str,
    *,
    model: str,
    api_key: Optional[str] = None,
    api_key_env: str = "OPENAI_API_KEY",
    base_url: str = "https://api.openai.com/v1",
) -> np.ndarray:
    """
    Minimal OpenAI-compatible embedding call (works for providers exposing /embeddings).
    """
    key = api_key or os.getenv(api_key_env)
    if not key:
        raise ValueError("Embedding API key missing; set env or pass api_key.")

    url = base_url.rstrip("/") + "/embeddings"
    payload = {"input": text, "model": model}
    resp = requests.post(
        url,
        json=payload,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    items = data.get("data") or []
    if not items or "embedding" not in items[0]:
        raise RuntimeError("Embedding response missing data.embedding")
    emb = np.array(items[0]["embedding"], dtype=np.float32)
    logger.info("Fetched embedding using base_url=%s model=%s input_len=%d", base_url, model, len(text))
    return emb


def hf_embedding(text: str, model_name: str) -> np.ndarray:
    raise NotImplementedError("HF embedding provider is a placeholder; add model loading before use.")


def build_embedding_from_config(config: Dict[str, Any]) -> Callable[[str], np.ndarray]:
    provider = config.get("provider", "random")
    if provider == "random":
        dim = config.get("dim", DEFAULT_DIM)
        return lambda text: random_embedding(text, dim=dim)
    if provider in {"openai", "openai_compat"}:
        model = config.get("model") or "text-embedding-3-small"
        api_key = config.get("api_key")
        api_key_env = config.get("api_key_env", "OPENAI_API_KEY")
        base_url = config.get("base_url", "https://api.openai.com/v1")
        return lambda text: openai_compatible_embedding(
            text,
            model=model,
            api_key=api_key,
            api_key_env=api_key_env,
            base_url=base_url,
        )
    if provider == "hf":
        model_name = config.get("model")
        if not model_name:
            raise ValueError("HF provider requires `model` in config.")
        return lambda text: hf_embedding(text, model_name=model_name)
    raise ValueError(f"Unknown embedding provider: {provider}")
