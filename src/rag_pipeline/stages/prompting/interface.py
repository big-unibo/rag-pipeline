from typing import Protocol

from rag_pipeline.models import Prompt, Question, RetrievedChunk


class PromptBuilder(Protocol):
    def build(self, question: Question, chunks: list[RetrievedChunk]) -> Prompt:
        ...
