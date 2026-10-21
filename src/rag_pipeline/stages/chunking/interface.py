from typing import Protocol

from rag_pipeline.models import Chunk, Document


class Chunker(Protocol):
    def chunk(self, documents: list[Document]) -> list[Chunk]:
        ...
