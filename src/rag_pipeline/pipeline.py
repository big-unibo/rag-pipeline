from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
import re
from time import perf_counter
from typing import Any, Callable, TypeVar

from sqlalchemy import create_engine, text
import yaml

from rag_pipeline.models import EmbeddedChunk, Evaluation, Question
from rag_pipeline.results import filter_evaluations
from rag_pipeline.settings import postgres_connection_string, resolve_path
from rag_pipeline.stages.retrieval import vector_literal
from rag_pipeline.stages.factory import build_stage


T = TypeVar("T")


class RagPipeline:
    def __init__(self, settings: dict[str, Any]):
        self.settings = settings
        self.engine = create_engine(postgres_connection_string(settings))

        # Each stage is discovered by class name from its configuration.
        self.document_loader = build_stage(
            settings.get("document_loading", {}),
            "PlainTextPdfLoader",
        )
        self.chunker = build_stage(settings.get("chunking", {}), "FixedChunker")
        self.embedder = build_stage(settings.get("embedding", {}), "OllamaEmbedder")
        self.retriever = build_stage(
            settings.get("retrieval", {}), "PostgresVectorRetriever", self.engine
        )
        self.graph_searcher = build_stage(
            settings.get("graph_search", {}),
            "Neo4jGraphSearcher",
            settings.get("credentials", {}),
        )
        self.reranker = build_stage(settings.get("ranking", {}), "OllamaReranker")
        self.prompt_builder = build_stage(settings.get("prompting", {}), "BasicPromptBuilder")
        self.answerer = build_stage(settings.get("answering", {}), "OllamaAnswerer")
        self.evaluator = build_stage(settings.get("evaluation", {}), "BasicEvaluator")

    def run_question(
        self,
        question: Question,
        position: int | None = None,
        total: int | None = None,
    ) -> Evaluation:
        # This is the online/query-time path:
        # embed question -> vector search + graph search -> rerank -> prompt LLM -> evaluate.
        question_label = (
            f"{position}/{total} {question.id}"
            if position is not None and total is not None
            else question.id
        )
        question_start = perf_counter()
        print(
            f"[question {question_label}] started at "
            f"{datetime.now().astimezone().isoformat(timespec='seconds')}",
            flush=True,
        )
        execution_times: dict[str, float] = {}
        embedded_question = self._timed_stage(
            question_label,
            "embedding",
            lambda: self.embedder.embed_question(question),
            execution_times,
        )
        vector_chunks = self._timed_stage(
            question_label,
            "retrieval",
            lambda: mark_source(self.retriever.retrieve(embedded_question), "vector"),
            execution_times,
        )
        graph_chunks = self._timed_stage(
            question_label,
            "graph_search",
            lambda: self.graph_searcher.search(embedded_question),
            execution_times,
        )
        ranked = self._timed_stage(
            question_label,
            "reranking",
            lambda: self.reranker.rank(
                embedded_question, deduplicate_chunks(vector_chunks + graph_chunks)
            ),
            execution_times,
        )
        prompt = self._timed_stage(
            question_label,
            "prompt",
            lambda: self.prompt_builder.build(embedded_question, ranked),
            execution_times,
        )
        answer = self._timed_stage(
            question_label,
            "answering",
            lambda: self.answerer.answer(prompt),
            execution_times,
        )
        metrics = self._timed_stage(
            question_label,
            "evaluation",
            lambda: self.evaluator.evaluate(embedded_question, answer),
            execution_times,
        )
        execution_times["total"] = perf_counter() - question_start
        evaluation = Evaluation(
            question_id=embedded_question.id,
            metrics=metrics,
            answer=answer.text,
            prompt=prompt.text,
            ground_truth=embedded_question.ground_truth,
            question_metadata=question.metadata,
            answer_metadata={
                "model": answer.model,
                "done_reason": answer.raw_response.get("done_reason", {}) if answer.raw_response else None,
            },
            retrieved_chunks=serialize_retrieved_chunks(ranked),
            execution_times=execution_times,
        )
        print(
            f"[question {question_label}] complete in {execution_times['total']:.2f}s",
            flush=True,
        )
        return evaluation

    @staticmethod
    def _timed_stage(
        question_id: str,
        stage: str,
        operation: Callable[[], T],
        execution_times: dict[str, float],
    ) -> T:
        start = perf_counter()
        try:
            return operation()
        finally:
            duration = perf_counter() - start
            execution_times[stage] = duration
            print(
                f"[question {question_id}] {stage} done {duration:.2f}s",
                flush=True,
            )

    def run(self) -> list[Evaluation]:
        # Questions are loaded from the selected corpus so the same pipeline
        # config can be reused across different document collections.
        questions = load_questions(self.settings, self.engine)
        total = len(questions)
        return [
            self.run_question(question, position, total)
            for position, question in enumerate(questions, start=1)
        ]

    def index_documents(self) -> int:
        # This is the offline/indexing path:
        # load documents -> split chunks -> embed chunks -> store vectors in Postgres.
        documents = self.document_loader.load_documents()
        chunks = self.chunker.chunk(documents)
        embedded_chunks = self.embedder.embed_chunks(chunks)
        store_embedded_chunks(self.settings, self.engine, embedded_chunks)
        return len(embedded_chunks)


def load_questions(settings: dict[str, Any], engine) -> list[Question]:
    question_config = settings.get("corpus", {}).get("questions", {})
    inline = question_config.get("inline")
    source = question_config.get("source")
    if inline is not None and source is not None:
        raise ValueError("Configure only one of corpus.questions.inline or corpus.questions.source.")

    schema = question_config.get("schema", {})
    metadata_config = question_config.get("metadata", {})
    if inline is not None:
        return parse_question_records(inline, schema, metadata_config, "<manifest inline>")

    source_kind = source.get("kind", "file")
    if source_kind == "database":
        records, source_name = load_database_question_source(source, schema, metadata_config, engine)
        return parse_question_records(records, schema, metadata_config, source_name)
    if source_kind != "file":
        raise ValueError(f"Unknown question source kind: {source_kind}")
    records, source_name = load_file_question_source(source)
    return parse_question_records(records, schema, metadata_config, source_name)


def load_file_question_source(source: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    path = source.get("path")
    if not path:
        raise ValueError("corpus.questions.source.path is required.")

    question_path = resolve_path(path)
    format_name = (source.get("format") or question_path.suffix.lstrip(".")).lower()
    with question_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle) if format_name == "json" else yaml.safe_load(handle)

    if isinstance(payload, dict):
        payload = payload.get("questions", [])
    if not isinstance(payload, list):
        raise ValueError(f"Question file {question_path} must contain a list or a questions list.")
    return payload, str(question_path)


def load_database_question_source(
    source: dict[str, Any],
    schema: dict[str, Any],
    metadata_config: dict[str, Any],
    engine,
) -> tuple[list[dict[str, Any]], str]:
    ids = source.get("ids", [])
    if not ids:
        raise ValueError("corpus.questions.source.ids must contain at least one question id.")

    table = validate_sql_identifier(source.get("table", "question"), "table")
    id_column = validate_database_field(schema.get("id", "q_id"), "schema.id")
    text_column = validate_database_field(schema.get("text", "question_text"), "schema.text")
    ground_truth_column = validate_database_field(
        schema.get("ground_truth", "ground_truth"),
        "schema.ground_truth",
    )
    metadata_fields = metadata_fields_from_config(metadata_config)
    metadata_columns = [validate_database_field(field, "questions.metadata.include") for field in metadata_fields]
    columns = list(dict.fromkeys([id_column, text_column, ground_truth_column, *metadata_columns]))
    select_columns = ", ".join(columns)
    query = text(
        f"""
        SELECT {select_columns}
        FROM {table}
        WHERE {id_column} = ANY(:question_ids)
        ORDER BY {id_column}
        """
    )
    with engine.connect() as connection:
        rows = connection.execute(query, {"question_ids": ids}).mappings().all()
    return [dict(row) for row in rows], f"database:{table}"


def validate_sql_identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"Invalid SQL identifier in {field_name}: {value!r}")
    return value


def validate_database_field(value: Any, field_name: str) -> str:
    if isinstance(value, list):
        raise ValueError(f"Database field {field_name} must be a single column name, not aliases.")
    return validate_sql_identifier(value, field_name)


def parse_question_records(
    records: list[dict[str, Any]],
    schema: dict[str, Any],
    metadata_config: dict[str, Any],
    source_name: str,
) -> list[Question]:
    if not isinstance(records, list):
        raise ValueError(f"Question source {source_name} must contain a list of records.")

    questions = []
    seen_ids: set[str] = set()
    for index, item in enumerate(records, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Question source {source_name} item {index} must be an object.")

        question_id = resolve_question_field(item, schema.get("id", "id"), "id", source_name, index, required=False)
        text_value = resolve_question_field(item, schema.get("text", "text"), "text", source_name, index)
        ground_truth = resolve_question_field(
            item,
            schema.get("ground_truth", "ground_truth"),
            "ground_truth",
            source_name,
            index,
            required=False,
        )
        question_id = str(question_id if question_id is not None else index)
        if question_id in seen_ids:
            raise ValueError(f"Duplicate question id {question_id!r} in {source_name} item {index}.")
        if not str(text_value).strip():
            raise ValueError(f"Question source {source_name} item {index} has empty text.")
        seen_ids.add(question_id)
        questions.append(
            Question(
                id=question_id,
                text=str(text_value).strip(),
                ground_truth=str(ground_truth).strip() if ground_truth else None,
                metadata=extract_question_metadata(item, metadata_config),
            )
        )
    return questions


def resolve_question_field(
    record: dict[str, Any],
    field_spec: str | list[str],
    field_name: str,
    source_name: str,
    index: int,
    required: bool = True,
) -> Any:
    candidates = [field_spec] if isinstance(field_spec, str) else field_spec
    for candidate in candidates:
        if candidate in record and record[candidate] is not None:
            return record[candidate]
    if required:
        raise ValueError(f"Question source {source_name} item {index} has no {field_name} field.")
    return None


def metadata_fields_from_config(metadata_config: dict[str, Any]) -> list[str]:
    if not isinstance(metadata_config, dict):
        raise ValueError("corpus.questions.metadata must be a mapping.")
    fields = metadata_config.get("include", [])
    if not isinstance(fields, list) or not all(isinstance(field, str) and field for field in fields):
        raise ValueError("corpus.questions.metadata.include must be a list of field names.")
    return fields


def extract_question_metadata(record: dict[str, Any], metadata_config: dict[str, Any]) -> dict[str, Any]:
    return {
        field: record[field]
        for field in metadata_fields_from_config(metadata_config)
        if field in record and record[field] is not None
    }

def serialize_evaluations(
    evaluations: list[Evaluation],
    ignore: list[str] | None = None,
) -> list[dict[str, Any]]:
    return filter_evaluations([asdict(evaluation) for evaluation in evaluations], ignore)


def serialize_retrieved_chunks(chunks) -> list[dict[str, Any]]:
    # Export retrieved chunks with each answer so students can debug retrieval
    # failures without stepping through the database manually.
    serialized = []
    for item in chunks:
        metadata = item.chunk.metadata
        serialized.append(
            {
                "rank": item.rank,
                "document_id": item.chunk.document_id,
                "chunk_id": metadata.get("chunk_id", item.chunk.id),
                "source": metadata.get("title"),
                "source_path": metadata.get("source_path"),
                "similarity": item.similarity,
                "score": item.score,
                "source_type": metadata.get("source_type") or item.metadata.get("source_type"),
                "text": item.chunk.text,
            }
        )
    return serialized


def mark_source(chunks, source_type: str):
    for item in chunks:
        item.metadata["source_type"] = source_type
        item.chunk.metadata["source_type"] = source_type
    return chunks


def deduplicate_chunks(chunks):
    # Vector and graph search can point to the same underlying chunk. Keep the
    # first copy because vector results arrive first and include similarity.
    seen = set()
    unique = []
    for item in chunks:
        key = (item.chunk.document_id, item.chunk.metadata.get("chunk_id", item.chunk.id))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def store_embedded_chunks(settings: dict[str, Any], engine, embedded_chunks: list[EmbeddedChunk]) -> None:
    retrieval_config = settings.get("retrieval", {})
    chunk_table = retrieval_config.get("chunk_table", "chunk_embeddings")
    document_table = retrieval_config.get("document_table", "documents")
    vector_column = retrieval_config.get("vector_column", "pgvector_embeddinggemma")
    chunking_model = retrieval_config.get("chunking_model", "structured")

    # Upserts make repeated experiments idempotent: changing prompt/evaluation
    # settings can reuse the existing index, while changing chunking_model writes
    # a separate set of chunks for comparison.
    document_sql = text(
        f"""
        INSERT INTO {document_table} (file_id, title, full_text)
        VALUES (:file_id, :title, :full_text)
        ON CONFLICT (file_id)
        DO UPDATE SET title = EXCLUDED.title, full_text = EXCLUDED.full_text
        """
    )
    chunk_sql = text(
        f"""
        INSERT INTO {chunk_table}
            (file_id, chunk_id, chunking_model, text, {vector_column})
        VALUES
            (:file_id, :chunk_id, :chunking_model, :chunk_text, CAST(:embedding AS vector))
        ON CONFLICT (file_id, chunking_model, chunk_id)
        DO UPDATE SET
            text = EXCLUDED.text,
            {vector_column} = EXCLUDED.{vector_column}
        """
    )

    documents: dict[str, dict[str, str]] = {}
    for item in embedded_chunks:
        chunk = item.chunk
        documents.setdefault(
            chunk.document_id,
            {
                "file_id": chunk.document_id,
                "title": chunk.metadata.get("title", chunk.document_id),
                "full_text": "",
            },
        )
        documents[chunk.document_id]["full_text"] += chunk.text + "\n\n"

    with engine.begin() as connection:
        for document in documents.values():
            connection.execute(document_sql, document)
        for item in embedded_chunks:
            chunk = item.chunk
            chunk_index = int(str(chunk.id).rsplit(":", 1)[-1])
            connection.execute(
                chunk_sql,
                {
                    "file_id": chunk.document_id,
                    "chunk_id": chunk_index,
                    "chunking_model": chunking_model,
                    "chunk_text": chunk.text,
                    "embedding": vector_literal(item.embedding),
                },
            )
