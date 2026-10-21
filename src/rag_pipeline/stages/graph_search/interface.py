from typing import Protocol

from rag_pipeline.models import Question, RetrievedChunk


class GraphSearcher(Protocol):
    def search(self, question: Question) -> list[RetrievedChunk]:
        ...
