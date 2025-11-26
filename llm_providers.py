import logging
import os
from typing import Any, Callable, Dict, Optional

import requests


def dummy_llm_fn(question: str, context: str, **_: Any) -> str:
    return f"[DUMMY ANSWER] Question: {question} | Context chars: {len(context)}"


def deepseek_chat_llm(
    question: str,
    context: str,
    *,
    api_key: Optional[str] = None,
    base_url: str = "https://api.deepseek.com",
    model: str = "deepseek-chat",
    temperature: float = 0.3,
    max_tokens: Optional[int] = 256,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Minimal DeepSeek chat completion helper using the OpenAI-compatible API.
    """
    key = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise ValueError("DeepSeek API key missing; set DEEPSEEK_API_KEY or pass api_key.")

    url = base_url.rstrip("/") + "/v1/chat/completions"
    messages = [
        {
            "role": "system",
            "content": system_prompt
            or "You are a concise QA assistant. Answer using the provided context; if unsure, say you do not know.",
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer succinctly.",
        },
    ]
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

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
    choices = data.get("choices")
    if not choices:
        raise RuntimeError("DeepSeek API returned no choices.")
    answer = choices[0]["message"]["content"].strip()
    logging.getLogger(__name__).info(
        "DeepSeek call success model=%s base_url=%s tokens_question=%d tokens_context=%d",
        model,
        base_url,
        len(question),
        len(context),
    )
    return answer


def build_llm_from_config(config: Dict[str, Any]) -> Callable[..., str]:
    provider = config.get("provider", "dummy")
    if provider == "dummy":
        return dummy_llm_fn
    if provider == "deepseek":
        base_url = config.get("base_url", "https://api.deepseek.com")
        model = config.get("model", "deepseek-chat")
        api_key = config.get("api_key")
        api_key_env = config.get("api_key_env", "DEEPSEEK_API_KEY")
        key_to_use = api_key or os.getenv(api_key_env)
        temperature = config.get("temperature", 0.3)
        max_tokens = config.get("max_tokens", 256)
        system_prompt = config.get("system_prompt")

        def _llm(question: str, context: str, **kwargs: Any) -> str:
            return deepseek_chat_llm(
                question=question,
                context=context,
                api_key=key_to_use,
                base_url=base_url,
                model=model,
                temperature=kwargs.get("temperature", temperature),
                max_tokens=kwargs.get("max_tokens", max_tokens),
                system_prompt=system_prompt,
            )

        return _llm
    raise ValueError(f"Unknown LLM provider: {provider}")
