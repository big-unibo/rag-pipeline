from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "pipeline.yml"
OUTPUTS = ROOT / "outputs"


def run(*args: str, expect_success: bool = True) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "-m", "rag_pipeline", *args]
    print(f"[e2e] {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=ROOT, text=True)
    if expect_success and result.returncode != 0:
        raise SystemExit(result.returncode)
    return result


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def assert_output_files() -> None:
    required = [
        OUTPUTS / "evaluation.json",
        OUTPUTS / "evaluation.csv",
        OUTPUTS / "retrieved_chunks.csv",
        OUTPUTS / "run_config.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise AssertionError(f"Missing pipeline outputs: {missing}")

    evaluations = csv_rows(OUTPUTS / "evaluation.csv")
    configs = csv_rows(OUTPUTS / "run_config.csv")
    if not evaluations:
        raise AssertionError("Pipeline produced no evaluation rows")
    config_ids = [row.get("run_config_id") for row in configs]
    if len(config_ids) != len(set(config_ids)):
        raise AssertionError("run_config.csv contains duplicate run configurations")
    if list(OUTPUTS.glob(".matrix-*")):
        raise AssertionError("Successful matrix run left a temporary directory")


def create_evaluator_input() -> Path:
    input_dir = OUTPUTS / "e2e-input"
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True)

    source_rows = csv_rows(OUTPUTS / "evaluation.csv")
    answer_fields = [
        "run_id",
        "run_approach",
        "question_id",
        "answer_text",
        "model",
        "prompt_text",
        "prompt_context",
        "execution_times",
    ]
    with (input_dir / "answers.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=answer_fields, delimiter=";")
        writer.writeheader()
        for source in source_rows:
            execution_times: dict[str, float] = {}
            for key, value in source.items():
                if not key.startswith("execution_time_") or not value:
                    continue
                name = key.removeprefix("execution_time_").removesuffix("_seconds")
                try:
                    execution_times[name] = float(value)
                except ValueError:
                    continue
            writer.writerow(
                {
                    "run_id": source.get("run_id", ""),
                    "run_approach": source.get("run_approach", ""),
                    "question_id": source.get("question_id", ""),
                    "answer_text": source.get("answer", ""),
                    "model": source.get("answer_metadata_model", ""),
                    "prompt_text": source.get("prompt", ""),
                    "prompt_context": "",
                    "execution_times": json.dumps(execution_times),
                }
            )
    shutil.copyfile(
        ROOT / "resources" / "datasets" / "medical" / "questions.csv",
        input_dir / "questions.csv",
    )
    shutil.copyfile(
        OUTPUTS / "retrieved_chunks.csv",
        input_dir / "chunks.csv",
    )
    return input_dir


def assert_evaluator_output(input_dir: Path) -> None:
    output_dir = OUTPUTS / "e2e-evaluation"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    run(
        "evaluate-csv",
        "--config",
        str(CONFIG),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--question-batch-size",
        "2",
    )
    output_csv = output_dir / "evaluation.csv"
    if not output_csv.exists() or not csv_rows(output_csv):
        raise AssertionError("Evaluator produced no evaluation rows")
    if list(output_dir.glob(".evaluator-*")):
        raise AssertionError("Successful evaluator run left a temporary directory")


def assert_failure_status(input_dir: Path) -> None:
    output_dir = OUTPUTS / "e2e-failure"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    result = run(
        "evaluate-csv",
        "--config",
        str(CONFIG),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--chunks",
        "missing-chunks.csv",
        expect_success=False,
    )
    if result.returncode == 0:
        raise AssertionError("Expected evaluator failure did not occur")
    temporary_runs = list(output_dir.glob(".evaluator-*/status.json"))
    if not temporary_runs:
        raise AssertionError("Failed evaluator did not retain a temporary run")
    statuses = [json.loads(path.read_text(encoding="utf-8")) for path in temporary_runs]
    if not any(status.get("status") == "failed" for status in statuses):
        raise AssertionError(f"No failed evaluator status found: {statuses}")


def main() -> None:
    run("run", "--config", str(CONFIG))
    run("matrix", "--config", str(CONFIG))
    assert_output_files()
    input_dir = create_evaluator_input()
    assert_evaluator_output(input_dir)
    assert_failure_status(input_dir)
    print("[e2e] all end-to-end checks passed", flush=True)


if __name__ == "__main__":
    main()
