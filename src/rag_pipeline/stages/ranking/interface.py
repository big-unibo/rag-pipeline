from typing import Protocol

from rag_pipeline.models import Question, RetrievedChunk


class Reranker(Protocol):
    def rank(self, question: Question, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        ...
