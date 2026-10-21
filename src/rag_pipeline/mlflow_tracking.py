from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

from rag_pipeline.models import Evaluation


@contextmanager
def tracked_run(settings: dict[str, Any]):
    tracking = settings.get("tracking", {})
    if not tracking.get("enabled", False):
        yield None
        return

    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("MLflow tracking is enabled, but mlflow is not installed.") from exc

    if tracking.get("tracking_uri"):
        mlflow.set_tracking_uri(tracking["tracking_uri"])
    mlflow.set_experiment(tracking.get("experiment_name", "rag-pipeline"))

    # One parent MLflow run is one matrix configuration. Long-lived stages also
    # create nested child runs so indexing can be filtered separately from the
    # question-time pipeline run in the MLflow UI.
    run_name = tracking.get("run_name")

    with mlflow.start_run(run_name=run_name):
        log_stage_params(settings)
        mlflow.set_tag("stage", "matrix")
        log_config_artifact(settings)
        yield mlflow


@contextmanager
def tracked_stage_run(stage: str, run_name: str | None = None):
    import mlflow

    child_name = f"{run_name}::{stage}" if run_name else stage
    with mlflow.start_run(run_name=child_name, nested=True):
        mlflow.set_tag("stage", stage)
        yield mlflow


def log_run(
    settings: dict[str, Any],
    evaluations: list[Evaluation],
    indexed_chunks: int | None = None,
    durations: dict[str, float] | None = None,
) -> None:
    with tracked_run(settings) as mlflow:
        if mlflow:
            log_run_results(evaluations, indexed_chunks, durations)


def log_run_results(
    evaluations: list[Evaluation],
    indexed_chunks: int | None = None,
    durations: dict[str, float] | None = None,
) -> None:
    import mlflow

    # Naming metrics with namespaces keeps the MLflow UI readable:
    # duration.*, index.*, question.*, and evaluation.*.
    if indexed_chunks is not None:
        mlflow.log_metric("index.indexed_chunks", indexed_chunks)
    for name, duration in (durations or {}).items():
        mlflow.log_metric(f"duration.{name}_seconds", duration)
    log_metrics(evaluations)


def log_stage_duration(stage: str, duration: float, extra_metrics: dict[str, float | int] | None = None) -> None:
    import mlflow

    mlflow.log_metric("stage.duration_seconds", duration)
    mlflow.log_metric(f"duration.{stage}_seconds", duration)
    for name, value in (extra_metrics or {}).items():
        mlflow.log_metric(name, value)


def log_stage_params(settings: dict[str, Any]) -> None:
    import mlflow

    # Log all configurable stage settings as params so two runs can be compared
    # directly in MLflow.
    for section in [
        "chunking",
        "embedding",
        "retrieval",
        "graph_search",
        "ranking",
        "prompting",
        "answering",
        "evaluation",
    ]:
        values = settings.get(section, {})
        if isinstance(values, dict):
            for key, value in flatten(values).items():
                if value is not None and not is_secret_key(key):
                    mlflow.log_param(f"{section}.{key}", value)

    matrix_run = settings.get("matrix_run", {})
    for key, value in flatten(matrix_run).items():
        mlflow.log_param(f"matrix.{key}", value)


def log_metrics(evaluations: list[Evaluation]) -> None:
    import mlflow

    if not evaluations:
        return

    mlflow.log_metric("evaluation.count", len(evaluations))
    metric_values: dict[str, list[float]] = {}
    for evaluation in evaluations:
        for key, value in evaluation.metrics.items():
            # MLflow metrics must be numeric. Strings such as judge rationales
            # stay in JSON/CSV outputs rather than metric charts.
            if isinstance(value, bool):
                metric_values.setdefault(key, []).append(1.0 if value else 0.0)
                mlflow.log_metric(f"question.{evaluation.question_id}.{key}", 1.0 if value else 0.0)
            elif isinstance(value, int | float):
                metric_values.setdefault(key, []).append(float(value))
                mlflow.log_metric(f"question.{evaluation.question_id}.{key}", float(value))

    for key, values in metric_values.items():
        mlflow.log_metric(f"evaluation.{key}.mean", sum(values) / len(values))


def log_config_artifact(settings: dict[str, Any]) -> None:
    import mlflow

    # Store the full effective config as an artifact. This captures matrix
    # overrides and environment overrides after they have been applied.
    redacted = redact(settings)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "pipeline_config.yml"
        path.write_text(yaml.safe_dump(redacted, sort_keys=True), encoding="utf-8")
        mlflow.log_artifact(str(path))


def flatten(values: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    # MLflow params are flat key/value pairs, but YAML configs are nested.
    flattened: dict[str, Any] = {}
    for key, value in values.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten(value, path))
        elif isinstance(value, list):
            flattened[path] = ",".join(str(item) for item in value)
        else:
            flattened[path] = value
    return flattened


def redact(value: Any) -> Any:
    # Keep experiment configs reproducible without leaking secrets into MLflow.
    if isinstance(value, dict):
        redacted = {}
        for key, nested in value.items():
            if is_secret_key(key):
                redacted[key] = "***"
            else:
                redacted[key] = redact(nested)
        return redacted
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def is_secret_key(key: str) -> bool:
    return any(secret in key.casefold() for secret in ["key", "password", "token", "secret"])
