from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Document:
    # Raw corpus item before chunking. A DocumentLoader can source PDFs, web
    # pages, database rows, Markdown files, or other document formats.
    id: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    # Small text unit used for retrieval. Chunk metadata is intentionally open
    # ended so experiments can add page numbers, section titles, layout data, etc.
    id: str
    document_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddedChunk:
    # Embeddings are numpy arrays while the pipeline is running. They are only
    # converted to pgvector's string format at the database boundary.
    chunk: Chunk
    embedding: np.ndarray


@dataclass(frozen=True)
class Question:
    # A question belongs to a corpus config. During retrieval we return a copy
    # with the embedding field populated by the selected Embedder.
    id: str
    text: str
    ground_truth: str | None = None
    embedding: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    # similarity comes from vector search; score can be filled by a later reranker.
    chunk: Chunk
    similarity: float | None = None
    score: float | None = None
    rank: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Prompt:
    # Keeping context and references beside the rendered text makes debugging
    # easier when comparing prompt templates.
    text: str
    context: str
    question: Question
    references: list[RetrievedChunk]
    messages: list[dict[str, str]] | None = None


@dataclass(frozen=True)
class Answer:
    # raw_response is model-specific. Keep it for debugging, but do not make
    # downstream stages depend on one vendor's response shape.
    text: str
    model: str
    prompt: Prompt
    raw_response: dict[str, Any] | None = None


@dataclass(frozen=True)
class Evaluation:
    # Each Evaluation is one question answered by one matrix run.
    question_id: str
    metrics: dict[str, Any]
    answer: str
    prompt: str
    ground_truth: str | None
    question_metadata: dict[str, Any] = field(default_factory=dict)
    answer_metadata: dict[str, Any] = field(default_factory=dict)
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)
    execution_times: dict[str, float] | None = field(default_factory=dict)
