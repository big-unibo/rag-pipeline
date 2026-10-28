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
from rag_pipeline.results import (
    append_retrieved_chunks_csv,
    append_results_csv,
    filter_results,
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
        reader = csv.DictReader(handle, delimiter=";")
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

def evaluate_csvs(
    input_dir: str | Path,
    output_dir: str | Path,
    questions_file: str = "questions.csv",
    answers_file: str = "answers.csv",
    chunks_file: str = "chunks.csv",
    evaluation_config: dict[str, Any] | None = None,
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

    evaluation_results: list[dict[str, Any]] = []
    for i, answer in enumerate(answers):
        question = question_by_id.get(answer["question_id"], {})
        run_id = answer["run_id"]
        approach = answer["run_approach"]
        run_chunks = chunks_by_run_question[(run_id, answer["question_id"])]
        execution_times = _json(answer.get("execution_times"))
        question_model = Question(
            id=answer["question_id"],
            text=question.get("text", ""),
            ground_truth=question.get("ground_truth") or None,
            metadata=_row_metadata(question, QUESTION_COLUMNS),
        )
        references = [
            # The same metadata object is used by Chunk and RetrievedChunk,
            # matching how metadata travels through the live pipeline.
            RetrievedChunk(
                chunk=Chunk(
                    id=chunk["chunk_id"],
                    document_id=chunk["document_id"],
                    text=chunk.get("text", ""),
                    metadata=chunk_metadata,
                ),
                similarity=_number(chunk.get("similarity")),
                score=_number(chunk.get("score")),
                rank=_number(chunk.get("rank")),
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
        print(f"Evaluating answer {i+1}/{len(answers)}: run_id={run_id}, approach={approach}, question_id={question_model.id}")
        evaluation_started = perf_counter()
        metrics = evaluator.evaluate(question_model, answer_model)
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
        evaluation_results.append({
            "run_id": run_id,
            "run_approach": approach,
            "evaluations": [asdict(evaluation_record)],
        })

    output_path.mkdir(parents=True, exist_ok=True)
    csv_ignore = (evaluation_config or {}).get("output", {}).get("csv", {}).get("ignore", [])
    detailed_path = append_results_csv(
        filter_results(evaluation_results, csv_ignore),
        output_path / "evaluation.csv",
    )
    chunks_path = append_retrieved_chunks_csv(
        evaluation_results,
        output_path / "retrieved_chunks.csv",
    )

    return {
        "detailed": detailed_path,
        "chunks": chunks_path
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
    )
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
