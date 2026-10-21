from typing import Protocol

import numpy as np

from rag_pipeline.models import Chunk, EmbeddedChunk, Question


class Embedder(Protocol):
    @property
    def dimension(self) -> int:
        ...

    def embed_text(self, text: str) -> np.ndarray:
        ...

    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        ...

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        ...

    def embed_question(self, question: Question) -> Question:
        ...
