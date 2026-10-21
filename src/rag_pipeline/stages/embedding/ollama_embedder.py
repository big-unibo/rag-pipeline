from __future__ import annotations

import numpy as np

from rag_pipeline.models import Chunk, EmbeddedChunk, Question
from rag_pipeline.ollama import OllamaClient


class OllamaEmbedder:
    def __init__(self, config: dict):
        self.ollama = OllamaClient(config, default_model="embeddinggemma")
        self.model = self.ollama.model
        self._dimension = int(config.get("dimension", 768))

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []

        embeddings = self.ollama.embed_texts(texts)
        if len(embeddings) != len(texts):
            raise RuntimeError(f"Ollama returned {len(embeddings)} embeddings for {len(texts)} inputs.")

        vectors = [np.asarray(embedding, dtype=float) for embedding in embeddings]
        for vector in vectors:
            if vector.shape != (self._dimension,):
                raise RuntimeError(
                    f"Ollama model {self.model} returned {vector.shape[0]} dimensions; "
                    f"expected {self._dimension}."
                )
        return vectors

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        vectors = self.embed_texts([chunk.text for chunk in chunks])
        return [EmbeddedChunk(chunk=chunk, embedding=vector) for chunk, vector in zip(chunks, vectors)]

    def embed_question(self, question: Question) -> Question:
        return Question(
            id=question.id,
            text=question.text,
            ground_truth=question.ground_truth,
            embedding=self.embed_text(question.text),
            metadata=question.metadata,
        )
