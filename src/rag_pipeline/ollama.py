"""Shared Ollama client and stage configuration helpers."""

from __future__ import annotations

from typing import Any

import requests


class OllamaClient:
    def __init__(self, config: dict[str, Any], default_model: str = "gemma4"):
        self.host = str(config.get("host", "http://ollama:11434")).rstrip("/")
        self.model = str(config.get("model", default_model))
        self.timeout = int(config.get("timeout", 600))
        self.keep_alive = config.get("keep_alive")
        self.session = requests.Session()

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0,
        num_predict: int | None = None,
        response_format: str | None = None,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {"temperature": temperature}
        if num_predict is not None:
            options["num_predict"] = num_predict
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if response_format is not None:
            payload["format"] = response_format
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        response = self.session.post(
            f"{self.host}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        response = self.session.post(
            f"{self.host}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload: dict[str, Any] = {"model": self.model, "input": texts}
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        response = self.session.post(
            f"{self.host}/api/embed",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json().get("embeddings", [])

    def pull(self, timeout: int = 1800) -> None:
        response = self.session.post(
            f"{self.host}/api/pull",
            json={"name": self.model, "stream": False},
            timeout=timeout,
        )
        response.raise_for_status()

OLLAMA_STAGE_STRATEGIES = frozenset(
    {
        "OllamaEmbedder",
        "OllamaAnswerer",
        "OllamaChatAnswerer",
        "OllamaReranker",
        "BasicEvaluator",
        "VragEvaluator",
        "TrustScoreEvaluator",
    }
)


def is_ollama_stage(value: Any) -> bool:
    """Return whether a stage configuration uses an Ollama implementation."""
    return isinstance(value, dict) and value.get("strategy", "") in OLLAMA_STAGE_STRATEGIES
