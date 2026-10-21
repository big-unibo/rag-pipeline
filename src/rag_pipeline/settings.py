from __future__ import annotations

from copy import deepcopy
from itertools import product
import os
from pathlib import Path
from typing import Any

import yaml

from rag_pipeline.ollama import is_ollama_stage


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "pipeline.yml"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    # Matrix options are small partial configs. Recursive merge lets an option
    # override only the fields it cares about, such as retrieval.top_k.
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_settings(config_path: str | Path = DEFAULT_CONFIG, corpus_path: str | Path | None = None) -> dict[str, Any]:
    config = load_yaml(config_path)
    settings = deep_merge(config, {"credentials": {}})

    # Keep algorithm choices in pipeline.yml and dataset choices in
    # resources/datasets/*/dataset.yml. This is the main teaching split in the project.
    load_corpus_settings(settings, corpus_path)
    settings["project_root"] = str(PROJECT_ROOT)
    apply_env_overrides(settings)
    return settings


def load_corpus_settings(settings: dict[str, Any], corpus_path: str | Path | None = None) -> None:
    # Priority order: environment variable for Docker/CI, CLI override for
    # ad-hoc runs, then the default path from config/pipeline.yml.
    selected_path = os.getenv("RAG_CORPUS_CONFIG") or corpus_path or settings.get("corpus", {}).get("path")
    if not selected_path:
        raise ValueError("Configure corpus.path in the pipeline config, pass --corpus, or set RAG_CORPUS_CONFIG.")

    corpus = load_yaml(resolve_path(selected_path))
    corpus["path"] = str(selected_path)
    settings["corpus"] = corpus

    documents = corpus.get("documents", {})
    if documents:
        # The loader selects how documents are read; the corpus owns their location.
        settings["document_loading"] = deep_merge(
            settings.get("document_loading", {}),
            documents,
        )


def expand_matrix(settings: dict[str, Any]) -> list[dict[str, Any]]:
    # The matrix is a Cartesian product of named options. For example, two
    # prompts and three retrievers create six full pipeline configurations.
    matrix = settings.get("matrix")
    if not matrix:
        return [settings]
    if not isinstance(matrix, dict):
        raise ValueError("matrix must be a mapping of dimension names to option lists.")

    dimensions: list[tuple[str, list[dict[str, Any]]]] = []
    for dimension_name, options in matrix.items():
        if not isinstance(options, list) or not options:
            raise ValueError(f"matrix.{dimension_name} must be a non-empty list.")
        dimensions.append((dimension_name, options))

    base = deepcopy(settings)
    base.pop("matrix", None)
    expanded: list[dict[str, Any]] = []

    for combination in product(*[options for _, options in dimensions]):
        run_settings = deepcopy(base)
        labels = []
        for dimension_name, option in zip([name for name, _ in dimensions], combination):
            option_name = option.get("name")
            if not option_name:
                raise ValueError(f"Every matrix option in {dimension_name} needs a name.")
            labels.append(f"{dimension_name}-{option_name}")
            run_settings = deep_merge(run_settings, option.get("overrides", {}))

        run_name = "__".join(labels)
        # Save option names separately from the raw config. CSV and MLflow use
        # this to group results by "prompt=strict-cited", "retrieval=top-5", etc.
        run_settings["matrix_run"] = {
            "name": run_name,
            "options": {
                dimension_name: option["name"]
                for dimension_name, option in zip([name for name, _ in dimensions], combination)
            },
        }
        tracking = run_settings.setdefault("tracking", {})
        base_run_name = tracking.get("run_name")
        tracking["run_name"] = f"{base_run_name}__{run_name}" if base_run_name else run_name
        apply_env_overrides(run_settings)
        expanded.append(run_settings)

    return expanded


def apply_env_overrides(settings: dict[str, Any]) -> None:
    # Docker Compose sets these variables so the same YAML also works from a
    # developer machine, CI, or inside the devcontainer.
    postgres = settings.setdefault("postgres", {})
    postgres["host"] = configured_or_env(postgres, "host", "POSTGRES_HOST", "localhost")
    postgres["port"] = int(configured_or_env(postgres, "port", "POSTGRES_PORT", 5432))
    postgres["database"] = configured_or_env(postgres, "database", "POSTGRES_DB", "rag")
    postgres["user"] = configured_or_env(postgres, "user", "POSTGRES_USER", "rag")
    postgres["password"] = configured_or_env(postgres, "password", "POSTGRES_PASSWORD", "rag")

    credentials = settings.setdefault("credentials", {})
    neo4j = credentials.setdefault("neo4j", {})
    neo4j["uri"] = configured_or_env(neo4j, "uri", "NEO4J_URI", "bolt://localhost:7687")
    neo4j["user"] = configured_or_env(neo4j, "user", "NEO4J_USER", "neo4j")
    neo4j["password"] = configured_or_env(neo4j, "password", "NEO4J_PASSWORD", "ragpassword")
    apply_neo4j_overrides(settings.get("graph_search"), neo4j)

    tracking = settings.setdefault("tracking", {})
    tracking["tracking_uri"] = configured_or_env(tracking, "tracking_uri", "MLFLOW_TRACKING_URI", None)
    tracking["experiment_name"] = configured_or_env(
        tracking,
        "experiment_name",
        "MLFLOW_EXPERIMENT_NAME",
        "rag-pipeline",
    )

    ollama_host = os.getenv("OLLAMA_HOST")
    for section in ["answering", "ranking", "evaluation", "embedding"]:
        apply_ollama_overrides(settings.get(section), ollama_host)


def apply_ollama_overrides(
    value: Any,
    host: str | None,
) -> None:
    # Walk nested config because Ollama settings can appear in multiple stages
    # and in the evaluator's nested embedding configuration.
    if isinstance(value, dict):
        if is_ollama_stage(value):
            if host and value.get("host") is None:
                value["host"] = host
        for nested in value.values():
            apply_ollama_overrides(nested, host)
    elif isinstance(value, list):
        for item in value:
            apply_ollama_overrides(item, host)


def apply_neo4j_overrides(value: Any, neo4j: dict[str, str]) -> None:
    if not isinstance(value, dict):
        return
    strategy = value.get("strategy", "")
    if strategy != "Neo4jGraphSearcher":
        return
    for key in ("uri", "user", "password"):
        if value.get(key) is None:
            value[key] = neo4j[key]


def configured_or_env(
    config: dict[str, Any],
    key: str,
    environment_key: str,
    default: Any,
) -> Any:
    """Resolve one setting using YAML, then environment, then a default."""
    configured = config.get(key)
    if configured is not None:
        return configured
    return os.getenv(environment_key, default)


def postgres_connection_string(settings: dict[str, Any]) -> str:
    postgres = settings["postgres"]
    return (
        f"postgresql+psycopg://{postgres['user']}:{postgres['password']}"
        f"@{postgres['host']}:{postgres['port']}/{postgres['database']}"
    )
