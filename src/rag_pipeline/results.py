from __future__ import annotations

import csv
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

def append_results_json(
    results: list[dict[str, Any]],
    json_path: str | Path,
    ignore: list[str] | None = None,
) -> Path:
    path = Path(json_path)
    existing = load_json_records(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(existing + filter_results(results, ignore), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path

def evaluation_field_preferred() -> list[str]:
    return [
        # Keep the most useful columns first; any new metric/config columns are
        # appended automatically so different runs can add fields safely.
        "run_id",
        "run_approach",
        "question_id",
        "prompt",
        "answer",
        "ground_truth",
        "indexed_chunks",
        "duration_index_seconds",
        "duration_run_seconds",
        "duration_total_seconds",
    ]


def run_config_field_preferred() -> list[str]:
    return [
        "run_config_id",
        "run_config_dataset",
        "run_config_document_loading_strategy",
        "run_config_chunking_strategy",
        "run_config_embedding_strategy",
        "run_config_embedding_model",
        "run_config_retrieval_strategy",
        "run_config_graph_search_strategy",
        "run_config_ranking_strategy",
        "run_config_ranking_model",
        "run_config_answering_strategy",
        "run_config_answering_model",
        "run_config_evaluation_strategy",
        "run_config_other_config_json",
    ]


def load_json_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return payload


def load_csv_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.exists() or path.stat().st_size == 0:
        return [], []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def ordered_fieldnames(
    existing_fields: list[str],
    rows: list[dict[str, Any]],
    preferred: list[str],
) -> list[str]:
    available = set(existing_fields)
    for row in rows:
        available.update(row)
    ordered = [field for field in preferred if field in available]
    ordered.extend(field for field in existing_fields if field not in ordered)
    ordered.extend(sorted(field for field in available if field not in ordered))
    return ordered


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_results_csv(
    results: list[dict[str, Any]],
    csv_path: str | Path,
    ignore: list[str] | None = None,
) -> Path:
    path = Path(csv_path)
    new_rows = flatten_results(filter_results(results, ignore))
    existing_rows, existing_fields = load_csv_rows(path)
    rows = existing_rows + new_rows
    fieldnames = ordered_fieldnames(existing_fields, rows, evaluation_field_preferred())
    write_csv_rows(path, rows, fieldnames)
    return path

def append_run_configs_csv(
    configs: list[dict[str, Any]],
    csv_path: str | Path,
) -> Path:
    path = Path(csv_path)
    new_rows = [flatten_mapping(config, "run_config") for config in configs]
    existing_rows, existing_fields = load_csv_rows(path)
    rows = existing_rows + new_rows
    fieldnames = ordered_fieldnames(existing_fields, rows, run_config_field_preferred())
    write_csv_rows(path, rows, fieldnames)
    return path


def append_retrieved_chunks_csv(
    results: list[dict[str, Any]],
    csv_path: str | Path,
) -> Path:
    path = Path(csv_path)
    new_rows = flatten_retrieved_chunks(results)
    existing_rows, existing_fields = load_csv_rows(path)
    rows = existing_rows + new_rows
    fieldnames = ordered_fieldnames(existing_fields, rows, retrieved_chunks_field_preferred())
    write_csv_rows(path, rows, fieldnames)
    return path


def retrieved_chunks_field_preferred() -> list[str]:
    return [
        "run_id",
        "question_id",
        "retrieved_rank",
        "document_id",
        "chunk_id",
        "source",
        "source_path",
        "source_type",
        "similarity",
        "score",
        "text",
    ]


def flatten_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Convert nested matrix output into one row per answered question.
    rows = []
    for result in results:
        base = {
            "run_id": result.get("run_id", result.get("run")),
            "run_approach": result.get("run_approach"),
            "indexed_chunks": result.get("indexed_chunks"),
        }
        base.update(flatten_mapping(result.get("run_config", {}), "run_config"))
        base.update(prefix(result.get("options", {}), "option"))
        base.update(prefix(result.get("durations_seconds", {}), "duration", suffix="_seconds"))

        for evaluation in result.get("evaluations", []):
            row = {**base}
            for field in ["question_id", "prompt", "answer", "ground_truth"]:
                if field in evaluation:
                    row[field] = evaluation[field]
            row.update(prefix(evaluation.get("metrics", {}), "metric"))
            row.update(
                prefix(evaluation.get("execution_times", {}), "execution_time", suffix="_seconds")
            )
            row.update(flatten_mapping(evaluation.get("question_metadata", {}), "question_metadata"))
            row.update(flatten_mapping(evaluation.get("answer_metadata", {}), "answer_metadata"))
            if "retrieved_chunks" in evaluation:
                row["retrieved_chunks"] = json.dumps(evaluation["retrieved_chunks"], ensure_ascii=False)
            rows.append(row)
    return rows


def flatten_retrieved_chunks(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Convert nested retrieved chunks into one row per question/chunk pair.
    rows = []
    for result in results:
        base = {
            "run_id": result.get("run_id", result.get("run")),
            "run_approach": result.get("run_approach"),
        }
        base.update(flatten_mapping(result.get("run_config", {}), "run_config"))
        base.update(prefix(result.get("options", {}), "option"))
        for evaluation in result.get("evaluations", []):
            for chunk in evaluation.get("retrieved_chunks", []):
                rows.append(
                    {
                        **base,
                        "question_id": evaluation.get("question_id"),
                        "retrieved_rank": chunk.get("rank"),
                        "document_id": chunk.get("document_id"),
                        "chunk_id": chunk.get("chunk_id"),
                        "source": chunk.get("source"),
                        "source_path": chunk.get("source_path"),
                        "source_type": chunk.get("source_type"),
                        "similarity": chunk.get("similarity"),
                        "score": chunk.get("score"),
                        "text": chunk.get("text"),
                    }
                )
    return rows


def prefix(values: dict[str, Any], name: str, suffix: str = "") -> dict[str, Any]:
    # Prefix avoids name collisions between matrix options, durations, metrics,
    # and ordinary result fields when everything is flattened into CSV.
    return {f"{name}_{key}{suffix}": value for key, value in values.items()}


def filter_results(results: list[dict[str, Any]], ignore: list[str] | None = None) -> list[dict[str, Any]]:
    """Return a copy with configured evaluation fields removed for one format."""
    filtered = deepcopy(results)
    for result in filtered:
        result["evaluations"] = filter_evaluations(result.get("evaluations", []), ignore)
    return filtered


def filter_evaluations(
    evaluations: list[dict[str, Any]],
    ignore: list[str] | None = None,
) -> list[dict[str, Any]]:
    ignored_fields = normalize_ignore_fields(ignore)
    filtered = deepcopy(evaluations)
    for evaluation in filtered:
        for field_path in ignored_fields:
            remove_field_path(evaluation, field_path)
    return filtered


def normalize_ignore_fields(ignore: list[str] | None) -> list[str]:
    if ignore is None:
        return []
    if not isinstance(ignore, list) or not all(isinstance(path, str) and path for path in ignore):
        raise ValueError("evaluation.output.<format>.ignore must be a list of field paths.")
    return ignore


def remove_field_path(value: dict[str, Any], field_path: str) -> None:
    current = value
    parts = field_path.split(".")
    for part in parts[:-1]:
        nested = current.get(part)
        if not isinstance(nested, dict):
            return
        current = nested
    current.pop(parts[-1], None)


def flatten_mapping(values: dict[str, Any], prefix_name: str) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in values.items():
        column = f"{prefix_name}_{key}"
        if isinstance(value, dict):
            flattened.update(flatten_mapping(value, column))
        elif isinstance(value, list):
            flattened[column] = json.dumps(value, ensure_ascii=False)
        else:
            flattened[column] = value
    return flattened
