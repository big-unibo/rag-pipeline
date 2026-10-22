from __future__ import annotations

import json
from copy import deepcopy
from time import perf_counter
import pandas as pd
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from rag_pipeline.mlflow_tracking import log_run_results, log_stage_duration, tracked_run, tracked_stage_run
from rag_pipeline.ollama import OllamaClient, is_ollama_stage
from rag_pipeline.pipeline import RagPipeline, serialize_evaluations
from rag_pipeline.results import (
    append_results_csv,
    append_results_json,
    append_retrieved_chunks_csv,
    append_run_configs_csv
)
from rag_pipeline.settings import expand_matrix, resolve_path


def run_matrix(settings: dict[str, Any]) -> list[dict[str, Any]]:
    runs = expand_matrix(settings)
    first_run_id = next_run_id(settings)
    for offset, run_settings in enumerate(runs):
        run_settings["run_id"] = first_run_id + offset
    for run_settings in runs:
        log_execution_config(run_settings)
    ensure_ollama_models(runs)

    # Indexing is usually the slowest stage. Cache by the settings that affect
    # stored chunks/vectors so prompt/evaluation variants do not re-index.
    indexed: dict[str, dict[str, float | int | None]] = {}
    should_index = bool(settings.get("execution", {}).get("index_documents", True))
    return [
        run_one_matrix_item(run, position, len(runs), should_index, indexed)
        for position, run in enumerate(runs, start=1)
    ]

def next_run_id(settings: dict[str, Any]) -> int:
    """Return the next persistent integer ID from the run-config history CSV."""
    execution = settings.get("execution", {})
    path = resolve_path(
        execution.get("output_run_config_csv", "outputs/run_config.csv")
    )

    if not path.exists() or path.stat().st_size == 0:
        return 1

    df = pd.read_csv(path)

    if "run_config_id" not in df.columns:
        raise ValueError(f"Expected a 'run_config_id' column in {path}")

    numeric_ids = pd.to_numeric(df["run_config_id"], errors="coerce").dropna()

    return int(numeric_ids.max()) + 1 if not numeric_ids.empty else 1


def log_execution_config(settings: dict[str, Any]) -> None:
    """Print a compact summary of the configured run and pipeline stages."""
    execution = settings.get("execution", {})
    corpus = settings.get("corpus", {})
    pipeline = " -> ".join(format_stages(settings))
    print(
        f"[config] "
        f"{settings.get('matrix_run', {}).get('name', settings.get('tracking', {}).get('run_name', 'default'))}"
        f" | corpus={corpus.get('name', corpus.get('path', 'default'))}"
        f" | index={'on' if execution.get('index_documents', True) else 'off'}"
        f" | {pipeline}"
    )


def format_stages(config: Any, path: str = "") -> list[str]:
    if isinstance(config, dict):
        stages = []
        if isinstance(config.get("strategy"), str):
            settings = ", ".join(
                f"{key}={format_setting(key, value)}"
                for key, value in config.items()
                if key != "strategy" and not isinstance(value, (dict, list))
                and not is_sensitive_setting(key)
            )
            stages.append(f"{path}={config['strategy']}({settings})")
        for key, value in config.items():
            if isinstance(value, (dict, list)):
                stages.extend(format_stages(value, f"{path}.{key}".strip(".")))
        return stages
    if isinstance(config, list):
        return [
            stage
            for index, value in enumerate(config)
            for stage in format_stages(value, f"{path}[{index}]")
        ]
    return []


def is_sensitive_setting(name: str) -> bool:
    return any(term in name.lower() for term in ("password", "secret", "token", "credential", "api_key", "user"))


def format_setting(name: str, value: Any) -> str:
    if name in {"host", "uri"} and isinstance(value, str):
        parsed = urlsplit(value)
        if parsed.scheme and parsed.hostname:
            host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
            try:
                port = parsed.port
            except ValueError:
                return "<configured>"
            netloc = f"{host}:{port}" if port is not None else host
            return urlunsplit((parsed.scheme, netloc, "", "", ""))
    return str(value)


def write_matrix_outputs(settings: dict[str, Any], results: list[dict[str, Any]]) -> None:
    # Append evaluation records and run configurations independently. JSON
    # preserves structure; CSV is easy to plot in Tableau or pandas.
    execution = settings.get("execution", {})
    output_config = settings.get("evaluation", {}).get("output", {})
    evaluation_results = [
        {key: value for key, value in result.items() if key != "run_config"}
        for result in results
    ]
    run_configs = [result["run_config"] for result in results if "run_config" in result]
    json_path = append_results_json(
        evaluation_results,
        execution.get("output_json", "outputs/evaluation.json"),
        output_config.get("json", {}).get("ignore", []),
    )
    csv_path = append_results_csv(
        evaluation_results,
        execution.get("output_csv", "outputs/evaluation.csv"),
        output_config.get("csv", {}).get("ignore", []),
    )
    run_config_csv_path = append_run_configs_csv(
        run_configs,
        execution.get("output_run_config_csv", "outputs/run_config.csv"),
    )
    chunks_path = append_retrieved_chunks_csv(
        evaluation_results,
        execution.get("output_retrieved_chunks_csv", "outputs/retrieved_chunks.csv"),
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {run_config_csv_path}")
    print(f"Wrote {chunks_path}")


def run_one_matrix_item(
    settings: dict[str, Any],
    position: int,
    total: int,
    should_index: bool,
    indexed: dict[str, dict[str, float | int | None]],
) -> dict[str, Any]:
    run_name = settings.get("matrix_run", {}).get("name", "default")
    key = indexing_key(settings)
    pipeline = RagPipeline(settings)

    # One MLflow run represents one full matrix variant. Stage durations are
    # logged as separate metrics to avoid duplicated experiment rows.
    with tracked_run(settings) as mlflow:
        total_start = perf_counter()
        if mlflow:
            with tracked_stage_run("indexing", run_name):
                indexed[key], index_duration = maybe_index(
                    pipeline,
                    settings,
                    key,
                    should_index,
                    indexed,
                    position,
                    total,
                    run_name,
                )
                log_stage_duration(
                    "indexing",
                    index_duration,
                    {
                        "index.indexed_chunks": indexed[key]["chunks"] or 0,
                        "index.performed": 1 if index_duration > 0 else 0,
                        "index.reused": 1 if should_index and index_duration == 0 else 0,
                        "index.skipped": 1 if not should_index else 0,
                    },
                )
        else:
            indexed[key], index_duration = maybe_index(
                pipeline,
                settings,
                key,
                should_index,
                indexed,
                position,
                total,
                run_name,
            )

        print(f"[{position}/{total}] Running pipeline for {run_name}...")
        if mlflow:
            with tracked_stage_run("question_run", run_name):
                run_start = perf_counter()
                evaluations = pipeline.run()
                run_duration = perf_counter() - run_start
                log_stage_duration("question_run", run_duration, {"evaluation.count": len(evaluations)})
        else:
            run_start = perf_counter()
            evaluations = pipeline.run()
            run_duration = perf_counter() - run_start

        durations = {
            "index": index_duration,
            "indexing": index_duration,
            "cached_index": indexed[key]["duration"],
            "run": run_duration,
            "question_run": run_duration,
            "total": perf_counter() - total_start,
        }
        if mlflow:
            log_run_results(evaluations, indexed_chunks=indexed[key]["chunks"], durations=durations)

    return {
        "run_approach": run_name,
        "run_id": settings.get("run_id", run_name),
        "run_config": build_run_config(settings, settings.get("run_id", 1)),
        "options": settings.get("matrix_run", {}).get("options", {}),
        "indexed_chunks": indexed[key]["chunks"],
        "durations_seconds": durations,
        "evaluations": serialize_evaluations(evaluations),
    }


def build_run_config(settings: dict[str, Any], run_id: int | str) -> dict[str, Any]:
    """Build the compact run configuration stored beside pipeline results."""
    corpus = settings.get("corpus", {})
    document_loading = settings.get("document_loading", {})
    chunking = settings.get("chunking", {})
    embedding = settings.get("embedding", {})
    retrieval = settings.get("retrieval", {})
    graph_search = settings.get("graph_search", {})
    ranking = settings.get("ranking", {})
    answering = settings.get("answering", {})
    evaluation = settings.get("evaluation", {})

    recurring = {
        "id": run_id,
        "dataset": corpus.get("name", corpus.get("path")),
        "document_loading_strategy": document_loading.get("strategy"),
        "chunking_strategy": chunking.get("strategy"),
        "embedding_strategy": embedding.get("strategy"),
        "embedding_model": embedding.get("model"),
        "retrieval_strategy": retrieval.get("strategy"),
        "graph_search_strategy": graph_search.get("strategy"),
        "ranking_strategy": ranking.get("strategy"),
        "ranking_model": ranking.get("model"),
        "answering_strategy": answering.get("strategy"),
        "answering_model": answering.get("model"),
        "evaluation_strategy": evaluation.get("strategy"),
    }

    other = deepcopy(settings)
    for section, fields in {
        "corpus": ("name", "path"),
        "document_loading": ("strategy",),
        "chunking": ("strategy",),
        "embedding": ("strategy", "model"),
        "retrieval": ("strategy",),
        "graph_search": ("strategy",),
        "ranking": ("strategy", "model"),
        "answering": ("strategy", "model"),
        "evaluation": ("strategy",),
    }.items():
        section_config = other.get(section)
        if isinstance(section_config, dict):
            for field in fields:
                section_config.pop(field, None)

    recurring["other_config_json"] = json.dumps(
        sanitize_config(other),
        sort_keys=True,
        default=str,
    )
    return recurring


def sanitize_config(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "<redacted>" if is_sensitive_setting(key) else sanitize_config(nested)
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [sanitize_config(item) for item in value]
    return value


def maybe_index(
    pipeline: RagPipeline,
    settings: dict[str, Any],
    key: str,
    should_index: bool,
    indexed: dict[str, dict[str, float | int | None]],
    position: int,
    total: int,
    run_name: str,
) -> tuple[dict[str, float | int | None], float]:
    if not should_index:
        # Use this mode when Postgres already contains the needed vectors.
        print(f"[{position}/{total}] Skipping indexing for {run_name}...")
        return {"chunks": None, "duration": 0.0}, 0.0
    if key in indexed:
        print(f"[{position}/{total}] Reusing index for {run_name}...")
        return indexed[key], 0.0

    print(f"[{position}/{total}] Indexing documents for {run_name}...")
    start = perf_counter()
    result = {"chunks": pipeline.index_documents(), "duration": perf_counter() - start}
    return result, result["duration"]


def indexing_key(settings: dict[str, Any]) -> str:
    retrieval = settings.get("retrieval", {})
    # Only include settings that change the persisted index. Prompt templates,
    # answer models, and evaluation metrics can safely reuse the same vectors.
    return json.dumps(
        {
            "chunking": settings.get("chunking", {}),
            "embedding": settings.get("embedding", {}),
            "chunk_table": retrieval.get("chunk_table"),
            "document_table": retrieval.get("document_table"),
            "vector_column": retrieval.get("vector_column"),
            "chunking_model": retrieval.get("chunking_model"),
        },
        sort_keys=True,
    )


def ensure_ollama_models(runs: list[dict[str, Any]]) -> None:
    # Pull every Ollama model once before the matrix starts. That makes the
    # timing of answer/evaluation runs less polluted by model download time.
    models: dict[str, set[str]] = {}
    for settings in runs:
        collect_ollama_models(settings, models)

    for host, model_names in models.items():
        for model in sorted(model_names):
            print(f"Ensuring Ollama model is available: {model}")
            OllamaClient({"host": host, "model": model}).pull()


def collect_ollama_models(value: Any, models: dict[str, set[str]]) -> None:
    # Recursively scan nested config because Ollama settings may appear in the
    # answerer and inside composite evaluation metrics.
    if isinstance(value, dict):
        if is_ollama_stage(value):
            host = value.get("host", "http://ollama:11434").rstrip("/")
            model = value.get("model", "gemma4")
            models.setdefault(host, set()).add(model)
        for nested in value.values():
            collect_ollama_models(nested, models)
    elif isinstance(value, list):
        for item in value:
            collect_ollama_models(item, models)
