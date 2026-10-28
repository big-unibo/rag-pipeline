from __future__ import annotations

from rag_pipeline.models import Answer, Prompt
from rag_pipeline.ollama import OllamaClient
from rag_pipeline.stages.answering.interface import Answerer


class OllamaAnswerer(Answerer):
    def __init__(self, config: dict):
        self.ollama = OllamaClient(config)
        self.model = self.ollama.model
        self.temperature = float(config.get("temperature", 0))
        self.num_predict = int(config.get("num_predict", 1024))

    def answer(self, prompt: Prompt) -> Answer:
        body = self.ollama.generate(
            prompt.text,
            temperature=self.temperature,
            num_predict=self.num_predict,
        )
        return Answer(text=body.get("response", ""), model=self.model, prompt=prompt, raw_response=body)
