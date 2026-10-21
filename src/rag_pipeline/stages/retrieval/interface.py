from typing import Protocol

from rag_pipeline.models import Question, RetrievedChunk


class Retriever(Protocol):
    def retrieve(self, question: Question) -> list[RetrievedChunk]:
        ...
