from __future__ import annotations

import csv
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any


class ResultBatchWriter:
    """Append completed result batches while a run is still in progress.

    Each call rewrites the target files with the existing rows plus the new
    batch. This keeps every output file readable if a later task fails or the
    process is interrupted.
    """

    def __init__(
        self,
        *,
        json_path: str | Path | None = None,
        csv_path: str | Path | None = None,
        chunks_csv_path: str | Path | None = None,
        run_config_csv_path: str | Path | None = None,
        json_ignore: list[str] | None = None,
        csv_ignore: list[str] | None = None,
    ) -> None:
        self.json_path = json_path
        self.csv_path = csv_path
        self.chunks_csv_path = chunks_csv_path
        self.run_config_csv_path = run_config_csv_path
        self.json_ignore = json_ignore or []
        self.csv_ignore = csv_ignore or []

    def write_batch(
        self,
        results: list[dict[str, Any]],
        run_configs: list[dict[str, Any]] | None = None,
    ) -> dict[str, Path]:
        """Persist one completed batch and return the files touched."""
        paths: dict[str, Path] = {}
        if self.json_path is not None:
            paths["json"] = append_results_json(results, self.json_path, self.json_ignore)
        if self.csv_path is not None:
            paths["detailed"] = append_results_csv(results, self.csv_path, self.csv_ignore)
        if self.chunks_csv_path is not None:
            paths["chunks"] = append_retrieved_chunks_csv(results, self.chunks_csv_path)
        if self.run_config_csv_path is not None and run_configs:
            paths["run_config"] = append_run_configs_csv(run_configs, self.run_config_csv_path)
        return paths


class StreamingResultBatchWriter:
    """Append checkpoints without rereading the complete temporary output."""

    def __init__(
        self,
        *,
        jsonl_path: str | Path | None = None,
        csv_path: str | Path | None = None,
        chunks_csv_path: str | Path | None = None,
        run_config_csv_path: str | Path | None = None,
        json_ignore: list[str] | None = None,
        csv_ignore: list[str] | None = None,
    ) -> None:
        self.jsonl_path = Path(jsonl_path) if jsonl_path is not None else None
        self.csv_path = Path(csv_path) if csv_path is not None else None
        self.chunks_csv_path = Path(chunks_csv_path) if chunks_csv_path is not None else None
        self.run_config_csv_path = (
            Path(run_config_csv_path) if run_config_csv_path is not None else None
        )
        self.json_ignore = json_ignore or []
        self.csv_ignore = csv_ignore or []

    def write_batch(
        self,
        results: list[dict[str, Any]],
        run_configs: list[dict[str, Any]] | None = None,
    ) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        json_results = filter_results(results, self.json_ignore)
        csv_results = filter_results(results, self.csv_ignore)
        if self.jsonl_path is not None:
            self._append_jsonl(json_results)
            paths["json"] = self.jsonl_path
        if self.csv_path is not None:
            self._append_csv(
                self.csv_path,
                flatten_results(csv_results),
                evaluation_field_preferred(),
            )
            paths["detailed"] = self.csv_path
        if self.chunks_csv_path is not None:
            self._append_csv(
                self.chunks_csv_path,
                flatten_retrieved_chunks(csv_results),
                retrieved_chunks_field_preferred(),
            )
            paths["chunks"] = self.chunks_csv_path
        if self.run_config_csv_path is not None and run_configs:
            self._append_csv(
                self.run_config_csv_path,
                [flatten_mapping(config, "run_config") for config in run_configs],
                run_config_field_preferred(),
            )
            paths["run_config"] = self.run_config_csv_path
        return paths

    def _append_jsonl(self, results: list[dict[str, Any]]) -> None:
        assert self.jsonl_path is not None
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    def _append_csv(
        self,
        path: Path,
        rows: list[dict[str, Any]],
        preferred: list[str],
    ) -> None:
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        existing_fields: list[str] = []
        if path.exists() and path.stat().st_size:
            with path.open(newline="", encoding="utf-8") as handle:
                existing_fields = list(csv.DictReader(handle).fieldnames or [])
        fieldnames = ordered_fieldnames(existing_fields, rows, preferred)
        if existing_fields and set(fieldnames) == set(existing_fields):
            with path.open("a", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=fieldnames).writerows(rows)
            return
        if not existing_fields:
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            return
        # Schema expansion is rare; rewrite only when a new metadata column appears.
        existing_rows, _ = load_csv_rows(path)
        write_csv_rows(path, existing_rows + rows, fieldnames)


class ResultWriter:
    """Public writer facade for final and streaming result output."""

    def __init__(self, *, streaming: bool = False, **kwargs: Any) -> None:
        writer_type = StreamingResultBatchWriter if streaming else ResultBatchWriter
        if streaming:
            kwargs["jsonl_path"] = kwargs.pop("json_path", kwargs.get("jsonl_path"))
        self._writer = writer_type(**kwargs)

    def write_batch(
        self,
        results: list[dict[str, Any]],
        run_configs: list[dict[str, Any]] | None = None,
    ) -> dict[str, Path]:
        return self._writer.write_batch(results, run_configs)


class TemporaryOutputSession:
    """Own a temporary run directory and its lifecycle status."""

    def __init__(self, parent: str | Path, prefix: str) -> None:
        self.parent = Path(parent)
        self.prefix = prefix
        self.path: Path | None = None

    def __enter__(self) -> "TemporaryOutputSession":
        self.parent.mkdir(parents=True, exist_ok=True)
        self.path = Path(tempfile.mkdtemp(prefix=self.prefix, dir=self.parent))
        self._write_status("running")
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: Any) -> bool:
        if self.path is None:
            return False
        if exc is not None:
            self._write_status("failed", f"{type(exc).__name__}: {exc}")
            return False
        self._write_status("completed")
        shutil.rmtree(self.path)
        return False

    def _write_status(self, status: str, error: str | None = None) -> None:
        if self.path is None:
            return
        payload: dict[str, Any] = {"status": status}
        if error:
            payload["error"] = error
        _atomic_write_text(self.path / "status.json", json.dumps(payload, indent=2, ensure_ascii=False))

def append_results_json(
    results: list[dict[str, Any]],
    json_path: str | Path,
    ignore: list[str] | None = None,
) -> Path:
    path = Path(json_path)
    existing = load_json_records(path)
    filtered = filter_results(results, ignore)
    def key_value(value: Any) -> str:
        return "" if value is None else str(value)

    seen = {
        (
            key_value(record.get("run_id", record.get("run"))),
            key_value(evaluation.get("question_id")),
        )
        for record in existing
        for evaluation in record.get("evaluations", [])
    }
    new_records: list[dict[str, Any]] = []
    for record in filtered:
        run_id = record.get("run_id", record.get("run"))
        normalized_run_id = key_value(run_id)
        evaluations = [
            evaluation
            for evaluation in record.get("evaluations", [])
            if (normalized_run_id, key_value(evaluation.get("question_id"))) not in seen
        ]
        if evaluations:
            new_records.append({**record, "evaluations": evaluations})
            seen.update(
                (normalized_run_id, key_value(evaluation.get("question_id")))
                for evaluation in evaluations
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        path,
        json.dumps(existing + new_records, indent=2, ensure_ascii=False),
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
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = handle.name
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            Path(temporary_path).unlink(missing_ok=True)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = handle.name
            handle.write(content)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            Path(temporary_path).unlink(missing_ok=True)


def append_only_new_rows(
    existing_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
    key_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Keep existing rows and append only rows with unseen business keys."""
    def normalize(value: Any) -> str:
        return "" if value is None else str(value)

    seen: set[tuple[str, ...]] = set()
    rows: list[dict[str, Any]] = []
    for row in existing_rows:
        key = tuple(normalize(row.get(field, "")) for field in key_fields)
        if any(not value for value in key) or key not in seen:
            rows.append(row)
            if all(key):
                seen.add(key)
    for row in new_rows:
        key = tuple(normalize(row.get(field, "")) for field in key_fields)
        if any(not value for value in key):
            rows.append(row)
            continue
        if key not in seen:
            rows.append(row)
            seen.add(key)
    return rows


def append_results_csv(
    results: list[dict[str, Any]],
    csv_path: str | Path,
    ignore: list[str] | None = None,
) -> Path:
    path = Path(csv_path)
    new_rows = flatten_results(filter_results(results, ignore))
    existing_rows, existing_fields = load_csv_rows(path)
    rows = append_only_new_rows(existing_rows, new_rows, ("run_id", "question_id"))
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
    rows = append_only_new_rows(existing_rows, new_rows, ("run_config_id",))
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
    rows = append_only_new_rows(
        existing_rows,
        new_rows,
        ("run_id", "question_id", "retrieved_rank"),
    )
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
