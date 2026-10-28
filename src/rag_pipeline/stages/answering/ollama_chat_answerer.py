from __future__ import annotations

from rag_pipeline.models import Answer, Prompt
from rag_pipeline.ollama import OllamaClient
from rag_pipeline.stages.answering.interface import Answerer


class OllamaChatAnswerer(Answerer):
    """Generate an answer from the message list built by a chat prompt stage."""

    def __init__(self, config: dict):
        self.ollama = OllamaClient(config)
        self.model = self.ollama.model
        self.temperature = float(config.get("temperature", 0))

    def answer(self, prompt: Prompt) -> Answer:
        if not prompt.messages:
            raise ValueError("OllamaChatAnswerer requires a prompt with messages.")

        body = self.ollama.chat(
            prompt.messages,
            temperature=self.temperature,
        )
        message = body.get("message", {})
        return Answer(
            text=message.get("content", ""),
            model=self.model,
            prompt=prompt,
            raw_response=body,
        )
