"""Combine and pivot evaluation CSV exports from different execution paths."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


def load_evaluation_csv(path: str | Path, source: str | None = None) -> pd.DataFrame:
    """Load one evaluation CSV and normalize its run identity columns."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Evaluation CSV does not exist: {csv_path}")
    delimiter = _detect_delimiter(csv_path)
    frame = pd.read_csv(
        csv_path,
        sep=delimiter,
        dtype={"run_id": "string", "run_approach": "string", "question_id": "string"},
    )
    if "run_approach" not in frame.columns:
        raise ValueError(f"{csv_path} must contain 'run_approach'.")
    for column in ("run_id", "question_id"):
        if column not in frame.columns:
            raise ValueError(f"{csv_path} must contain '{column}'.")
    frame["run_approach"] = frame["run_approach"].fillna("").astype("string")
    frame["question_id"] = frame["question_id"].astype("string")
    frame["result_source"] = source or csv_path.stem
    return frame


def concat_evaluations(
    pipeline_csv: str | Path,
    evaluator_csv: str | Path,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Concatenate standard-pipeline and offline-evaluator results."""
    pipeline = load_evaluation_csv(pipeline_csv, "pipeline")
    evaluator = load_evaluation_csv(evaluator_csv, "evaluate_answers")
    combined = pd.concat([pipeline, evaluator], ignore_index=True, sort=False)
    if output_csv is not None:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_path, index=False)
    return combined


def pivot_evaluation(
    evaluation_csv: str | Path | pd.DataFrame,
    output_csv: str | Path | None = None,
    *,
    index_columns: Iterable[str] = ("question_id",),
    value_columns: Iterable[str] | None = None,
    aggregation: str = "first",
) -> pd.DataFrame:
    """Pivot metric and timing values into columns grouped by run approach."""
    frame = (
        evaluation_csv.copy()
        if isinstance(evaluation_csv, pd.DataFrame)
        else load_evaluation_csv(evaluation_csv)
    )
    indexes = list(index_columns)
    missing = [column for column in [*indexes, "run_approach"] if column not in frame.columns]
    if missing:
        raise ValueError(f"Evaluation data is missing pivot columns: {', '.join(missing)}")

    values = list(value_columns) if value_columns is not None else [
        column for column in frame.columns
        if column.startswith(("metric_", "execution_time_", "duration_", "answer_metadata_"))
    ]
    values = [column for column in values if column in frame.columns]
    if not values:
        raise ValueError("No metric or timing columns available to pivot.")

    pivoted = frame.pivot_table(
        index=indexes,
        columns="run_approach",
        values=values,
        aggfunc=aggregation,
    )
    pivoted.columns = [f"{metric}__{approach}" for metric, approach in pivoted.columns]
    pivoted = pivoted.reset_index()
    if output_csv is not None:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pivoted.to_csv(output_path, index=False)
    return pivoted


def _detect_delimiter(path: Path) -> str:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if not lines:
        raise ValueError(f"Evaluation CSV is empty: {path}")
    first_line = lines[0]
    return ";" if first_line.count(";") > first_line.count(",") else ","


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Combine and pivot RAG evaluation CSVs.")
    parser.add_argument("--pipeline", default="outputs/evaluation_medical_full.csv", help="Standard pipeline evaluation CSV.")
    parser.add_argument("--evaluator", default="outputs/comparison/evaluation.csv", help="evaluate_answers evaluation CSV.")
    parser.add_argument("--output", default="outputs/comparison_global/combined_evaluation.csv")
    parser.add_argument("--pivot-output", default="outputs/comparison_global/evaluation_pivot.csv")
    args = parser.parse_args(argv)
    combined = concat_evaluations(args.pipeline, args.evaluator, args.output)
    pivot_evaluation(combined, args.pivot_output)
    print(f"Wrote combined evaluations to {args.output}")
    print(f"Wrote pivoted evaluations to {args.pivot_output}")


if __name__ == "__main__":
    main()
