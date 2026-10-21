from __future__ import annotations

import numpy as np
from sqlalchemy import text
from sqlalchemy.engine import Engine

from rag_pipeline.models import Chunk, Question, RetrievedChunk


def vector_literal(vector: np.ndarray) -> str:
    # psycopg does not know pgvector by default in this tiny teaching setup, so
    # we send vectors in pgvector's text format: "[0.1,0.2,...]".
    values = np.asarray(vector, dtype=float)
    return "[" + ",".join(str(value) for value in values.tolist()) + "]"


class PostgresVectorRetriever:
    def __init__(self, config: dict, engine: Engine):
        self.engine = engine
        self.top_k = int(config.get("top_k", 5))
        self.threshold = config.get("threshold")
        self.chunk_table = config.get("chunk_table", "chunk_embeddings")
        self.document_table = config.get("document_table", "documents")
        self.vector_column = config.get("vector_column", "pgvector_embeddinggemma")
        self.text_column = config.get("text_column", "text")
        self.chunking_model = config.get("chunking_model", "structured")

    def retrieve(self, question: Question) -> list[RetrievedChunk]:
        if question.embedding is None:
            raise ValueError("Question must be embedded before retrieval.")

        similarity_filter = ""
        params = {
            "query_vector": vector_literal(question.embedding),
            "top_k": self.top_k,
            "chunking_model": self.chunking_model,
        }
        if self.threshold is not None:
            # Lower cosine similarity means more similar. threshold can cut noisy
            # results before prompt construction.
            similarity_filter = f"AND (c.{self.vector_column} <=> CAST(:query_vector AS vector)) >= :threshold"
            params["threshold"] = self.threshold

        query = text(
            f"""
            SELECT
                d.title AS title,
                d.file_id AS file_id,
                c.chunk_id AS chunk_id,
                c.{self.text_column} AS chunk_text,
                1 - (c.{self.vector_column} <=> CAST(:query_vector AS vector)) AS similarity
            FROM {self.chunk_table} c
            JOIN {self.document_table} d ON c.file_id = d.file_id
            WHERE c.chunking_model = :chunking_model
            AND c.{self.vector_column} IS NOT NULL
            {similarity_filter}
            ORDER BY similarity DESC
            LIMIT :top_k
            """
        )

        with self.engine.connect() as connection:
            rows = connection.execute(query, params).fetchall()

        results: list[RetrievedChunk] = []
        for rank, row in enumerate(rows, start=1):
            chunk = Chunk(
                id=f"{row.file_id}:{row.chunk_id}",
                document_id=str(row.file_id),
                text=row.chunk_text,
                metadata={
                    "title": row.title,
                    "file_id": row.file_id,
                    "chunk_id": row.chunk_id,
                },
            )
            results.append(RetrievedChunk(chunk=chunk, similarity=float(row.similarity), rank=rank))
        return results
