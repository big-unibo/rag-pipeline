from typing import Protocol

from rag_pipeline.models import Answer, Prompt


class Answerer(Protocol):
    def answer(self, prompt: Prompt) -> Answer:
        ...
