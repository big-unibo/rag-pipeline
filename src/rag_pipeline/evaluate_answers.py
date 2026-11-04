"""Evaluate exported question, answer, and retrieved-chunk CSV files.

This module reuses the same models and configured evaluator stage as the live
pipeline, but operates on already completed runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

from rag_pipeline.models import Answer, Chunk, Evaluation, Prompt, Question, RetrievedChunk
from rag_pipeline.execution import execute_batched
from rag_pipeline.results import (
    ResultWriter,
    TemporaryOutputSession,
)
from rag_pipeline.stages.factory import build_stage

QUESTION_COLUMNS = {"id", "text", "ground_truth"}
ANSWER_COLUMNS = {
    "run_id", "run_approach", "question_id", "answer_text",
    "model", "prompt_text", "prompt_context", "execution_times"
}
CHUNK_COLUMNS = {
    "run_id", "run_approach", "question_id", "document_id", "chunk_id",
    "similarity", "score", "text"
}


def _read_csv(path: Path, expected: set[str]) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        first_line = handle.readline()
        delimiter = ";" if first_line.count(";") > first_line.count(",") else ","
        handle.seek(0)
        reader = csv.DictReader(handle, delimiter=delimiter)
        columns = set(reader.fieldnames or [])
        missing = expected - columns
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
        return [dict(row) for row in reader]


def _json(value: str | None) -> Any:
    if not value:
        return {}
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}


def _row_metadata(row: dict[str, str], known_columns: set[str]) -> dict[str, Any]:
    """Collect arbitrary non-core CSV columns as model metadata."""
    return {
        key: value
        for key, value in row.items()
        if key not in known_columns
    }


def _number(value: str | None) -> float | None:
    try:
        number = float(value) if value not in (None, "") else None
        return number if number is not None and math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _rank(value: str | None) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def _chunk_rank(chunk: dict[str, str]) -> int | None:
    """Accept both evaluator ``rank`` and pipeline ``retrieved_rank`` names."""
    return _rank(chunk.get("rank") or chunk.get("retrieved_rank"))


def _evaluate_answer(
    answer: dict[str, str],
    position: int,
    total: int,
    question_by_id: dict[str, dict[str, str]],
    chunks_by_run_question: dict[tuple[str, str], list[dict[str, str]]],
    evaluator: Any,
) -> dict[str, Any]:
    question = question_by_id.get(answer["question_id"], {})
    run_id = answer["run_id"]
    approach = answer["run_approach"]
    run_chunks = sorted(
        chunks_by_run_question[(run_id, answer["question_id"])],
        key=lambda chunk: _chunk_rank(chunk)
        if _chunk_rank(chunk) is not None
        else float("inf"),
    )
    execution_times = _json(answer.get("execution_times"))
    question_model = Question(
        id=answer["question_id"],
        text=question.get("text", ""),
        ground_truth=question.get("ground_truth") or None,
        metadata=_row_metadata(question, QUESTION_COLUMNS),
    )
    references = [
        RetrievedChunk(
            chunk=Chunk(
                id=chunk["chunk_id"],
                document_id=chunk["document_id"],
                text=chunk.get("text", ""),
                metadata=chunk_metadata,
            ),
            similarity=_number(chunk.get("similarity")),
            score=_number(chunk.get("score")),
            rank=_chunk_rank(chunk),
        )
        for chunk in run_chunks
        for chunk_metadata in [_row_metadata(chunk, CHUNK_COLUMNS)]
    ]
    prompt = Prompt(
        text=answer.get("prompt_text", ""),
        context=answer.get("prompt_context", ""),
        question=question_model,
        references=references,
    )
    answer_model = Answer(
        text=answer.get("answer_text", ""),
        model=answer.get("model", ""),
        prompt=prompt,
        raw_response={},
    )
    print(
        f"Evaluating answer {position}/{total}: run_id={run_id}, "
        f"approach={approach}, question_id={question_model.id}"
    )
    evaluation_started = perf_counter()
    try:
        metrics = evaluator.evaluate(question_model, answer_model)
    except Exception as exc:  # preserve the failed task in the report
        metrics = {
            "evaluation_status": "failed",
            "evaluation_error": f"{type(exc).__name__}: {exc}",
        }
        print(
            f"[question {question_model.id}] evaluation failed: {metrics['evaluation_error']}",
            flush=True,
        )
    else:
        metrics = {"evaluation_status": "completed", **metrics}
    evaluation_time = perf_counter() - evaluation_started
    evaluation_record = Evaluation(
        question_id=question_model.id,
        metrics=metrics,
        answer=answer_model.text,
        prompt=prompt.text,
        ground_truth=question_model.ground_truth,
        question_metadata=question_model.metadata,
        answer_metadata={**_row_metadata(answer, ANSWER_COLUMNS), "model": answer_model.model},
        retrieved_chunks=[
            {
                "rank": reference.rank,
                "document_id": reference.chunk.document_id,
                "chunk_id": reference.chunk.id,
                "similarity": reference.similarity,
                "score": reference.score,
                "text": reference.chunk.text,
            }
            for reference in references
        ],
        execution_times={
            **(execution_times if isinstance(execution_times, dict) else {}),
            "evaluation": evaluation_time,
        },
    )
    print(
        f"[question {question_model.id}] complete evaluation in {evaluation_time:.2f}s",
        flush=True,
    )
    return {
        "run_id": run_id,
        "run_approach": approach,
        "evaluations": [asdict(evaluation_record)],
    }

def evaluate_csvs(
    input_dir: str | Path,
    output_dir: str | Path,
    questions_file: str = "questions.csv",
    answers_file: str = "answers.csv",
    chunks_file: str = "chunks.csv",
    evaluation_config: dict[str, Any] | None = None,
    question_batch_size: int | None = None,
) -> dict[str, Path]:
    """Evaluate CSV exports and retain failed temporary runs for recovery."""
    output_path = Path(output_dir)
    with TemporaryOutputSession(output_path, ".evaluator-") as session:
        result = _evaluate_csvs_impl(
            input_dir,
            output_path,
            questions_file,
            answers_file,
            chunks_file,
            evaluation_config,
            question_batch_size,
            session.path,
        )
        return result


def _evaluate_csvs_impl(
    input_dir: str | Path,
    output_dir: str | Path,
    questions_file: str = "questions.csv",
    answers_file: str = "answers.csv",
    chunks_file: str = "chunks.csv",
    evaluation_config: dict[str, Any] | None = None,
    question_batch_size: int | None = None,
    temporary_dir: Path | None = None,
) -> dict[str, Path]:
    """Evaluate CSV exports through the configured pipeline evaluation stage."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    questions = _read_csv(input_path / questions_file, QUESTION_COLUMNS)
    answers = _read_csv(input_path / answers_file, ANSWER_COLUMNS)
    chunks = _read_csv(input_path / chunks_file, CHUNK_COLUMNS)
    evaluator = build_stage(evaluation_config or {}, "BasicEvaluator")

    question_by_id = {row["id"]: row for row in questions}
    chunks_by_run_question: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for chunk in chunks:
        key = (chunk["run_id"], chunk["question_id"])
        chunks_by_run_question[key].append(chunk)

    if temporary_dir is None:
        raise ValueError("temporary_dir is required for evaluator execution")
    output_settings = (evaluation_config or {}).get("output", {})
    effective_question_batch_size = question_batch_size if question_batch_size is not None else 10
    csv_ignore = output_settings.get("csv", {}).get("ignore", [])
    writer = ResultWriter(
        streaming=True,
        jsonl_path=temporary_dir / "evaluation.jsonl",
        csv_path=temporary_dir / "evaluation.csv",
        chunks_csv_path=temporary_dir / "retrieved_chunks.csv",
        csv_ignore=csv_ignore,
    )
    def write_temporary_batch(batch: list[dict[str, Any]]) -> None:
        writer.write_batch(batch)

    all_evaluation_results = execute_batched(
        answers,
        lambda answer, position, total: _evaluate_answer(
            answer,
            position,
            total,
            question_by_id,
            chunks_by_run_question,
            evaluator,
        ),
        effective_question_batch_size,
        write_temporary_batch,
    )

    # Persist complete results only after every answer has been processed.
    final_writer = ResultWriter(
        csv_path=output_path / "evaluation.csv",
        chunks_csv_path=output_path / "retrieved_chunks.csv",
        csv_ignore=csv_ignore,
    )
    final_writer.write_batch(all_evaluation_results)
    return {
        "detailed": output_path / "evaluation.csv",
        "chunks": output_path / "retrieved_chunks.csv",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate exported question, answer, and retrieved-chunk CSV files."
    )
    parser.add_argument("--config", default="config/pipeline.develop.yml", help="Pipeline YAML config.")
    parser.add_argument("--input-dir", default="outputs/input", help="Directory containing input CSVs.")
    parser.add_argument("--output-dir", default="outputs/comparison", help="Directory for evaluation output.")
    parser.add_argument("--questions", default="questions.csv", help="Questions CSV filename.")
    parser.add_argument("--answers", default="answers.csv", help="Answers CSV filename.")
    parser.add_argument("--chunks", default="chunks.csv", help="Retrieved chunks CSV filename.")
    parser.add_argument("--question-batch-size", type=int, default=None, help="Flush output after this many questions.")
    args = parser.parse_args(argv)

    from rag_pipeline.settings import load_yaml

    settings = load_yaml(args.config)
    print(settings.get("evaluation", {}))
    print(f"Evaluating CSVs in {args.input_dir} and writing results to {args.output_dir}")
    paths = evaluate_csvs(
        args.input_dir,
        args.output_dir,
        args.questions,
        args.answers,
        args.chunks,
        settings.get("evaluation", {}),
        question_batch_size=(
            args.question_batch_size
            if args.question_batch_size is not None
            else settings.get("execution", {}).get("question_batch_size")
        ),
    )
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
