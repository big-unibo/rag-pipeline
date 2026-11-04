from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_CONFIG = "config/pipeline.yml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the configurable RAG pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run question answering and evaluation.")
    run_parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to a pipeline YAML config.")
    run_parser.add_argument("--corpus", "--dataset", dest="corpus", default=None, help="Optional corpus YAML override.")
    run_parser.add_argument("--output", default=None, help="Optional JSON output path for evaluations.")

    index_parser = subparsers.add_parser("index", help="Chunk, embed, and store local documents.")
    index_parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to a pipeline YAML config.")
    index_parser.add_argument("--corpus", "--dataset", dest="corpus", default=None, help="Optional corpus YAML override.")

    matrix_parser = subparsers.add_parser("matrix", help="Run every configured matrix combination.")
    matrix_parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to a pipeline YAML config.")
    matrix_parser.add_argument("--corpus", "--dataset", dest="corpus", default=None, help="Optional corpus YAML override.")
    matrix_parser.add_argument("--output", default=None, help="JSON output path.")
    matrix_parser.add_argument("--csv", default=None, help="CSV output path.")
    matrix_parser.add_argument("--chunks-csv", default=None, help="Retrieved chunks CSV path.")
    matrix_parser.add_argument("--question-batch-size", type=int, default=None, help="Flush output after this many questions.")
    matrix_parser.add_argument("--skip-index", action="store_true", help="Use chunks already available in Postgres.")

    csv_parser = subparsers.add_parser(
        "evaluate-csv", help="Evaluate exported questions, answers, and chunks CSVs."
    )
    csv_parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Pipeline YAML containing the evaluation stage configuration.")
    csv_parser.add_argument("--input-dir", default="outputs/input", help="Directory containing the three input CSVs.")
    csv_parser.add_argument("--output-dir", default="outputs/comparison", help="Dedicated evaluation output directory.")
    csv_parser.add_argument("--questions", default="questions.csv")
    csv_parser.add_argument("--answers", default="answers.csv")
    csv_parser.add_argument("--chunks", default="chunks.csv")
    csv_parser.add_argument("--question-batch-size", type=int, default=None, help="Flush evaluator output after this many questions.")

    compare_parser = subparsers.add_parser(
        "compare-evaluations", help="Combine and pivot pipeline evaluation CSVs."
    )
    compare_parser.add_argument("--pipeline", required=True, help="Standard pipeline evaluation CSV.")
    compare_parser.add_argument("--evaluator", required=True, help="evaluate_answers evaluation CSV.")
    compare_parser.add_argument("--output", default="outputs/comparison/combined_evaluation.csv")
    compare_parser.add_argument("--pivot-output", default="outputs/comparison/evaluation_pivot.csv")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "evaluate-csv":
        from rag_pipeline.evaluate_answers import evaluate_csvs
        from rag_pipeline.settings import load_yaml

        settings = load_yaml(args.config)
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
                else settings.get("execution", {}).get("question_batch_size", 10)
            ),
        )
        print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))
        return

    if args.command == "compare-evaluations":
        from rag_pipeline.compare_evaluations import concat_evaluations, pivot_evaluation

        combined = concat_evaluations(args.pipeline, args.evaluator, args.output)
        pivot_evaluation(combined, args.pivot_output)
        print(json.dumps({"combined": args.output, "pivot": args.pivot_output}, indent=2))
        return

    # Import runtime dependencies after argument parsing so "rag-pipeline --help"
    # stays fast and does not load ML/DB libraries.
    from rag_pipeline.pipeline import RagPipeline, serialize_evaluations
    from rag_pipeline.mlflow_tracking import log_run
    from rag_pipeline.results import filter_results
    from rag_pipeline.runner import log_execution_config, run_matrix, run_single_pipeline
    from rag_pipeline.settings import load_settings

    settings = load_settings(args.config, args.corpus)

    if args.command == "matrix":
        if args.skip_index:
            settings.setdefault("execution", {})["index_documents"] = False
        # Matrix mode owns its question-batch writer. The CLI only applies
        # optional path and batch-size overrides before starting execution.
        execution = settings.get("execution", {})
        if args.output is not None:
            execution["output_json"] = args.output
        if args.csv is not None:
            execution["output_csv"] = args.csv
        if args.chunks_csv is not None:
            execution["output_retrieved_chunks_csv"] = args.chunks_csv
        question_batch_size = (
            args.question_batch_size
            if args.question_batch_size is not None
            else execution.get("question_batch_size", 10)
        )

        results = run_matrix(
            settings,
            question_batch_size=question_batch_size,
        )
        json_ignore = settings.get("evaluation", {}).get("output", {}).get("json", {}).get("ignore", [])
        print(json.dumps(filter_results(results, json_ignore), indent=2, ensure_ascii=False))
        return

    log_execution_config(settings)

    if args.command == "index":
        # Index-only mode is useful before running many LLM or prompt variants
        # against the same Postgres data.
        pipeline = RagPipeline(settings)
        indexed = pipeline.index_documents()
        print(json.dumps({"indexed_chunks": indexed}, indent=2))
        return

    evaluations, _ = run_single_pipeline(settings)
    log_run(settings, evaluations)
    output_config = settings.get("evaluation", {}).get("output", {})
    payload = serialize_evaluations(evaluations, output_config.get("json", {}).get("ignore", []))

    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    print(rendered)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
