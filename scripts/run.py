from __future__ import annotations

from rag_pipeline.runner import run_matrix, write_matrix_outputs
from rag_pipeline.settings import load_settings


def main() -> None:
    settings = load_settings("config/pipeline.yml")
    write_matrix_outputs(settings, run_matrix(settings))


if __name__ == "__main__":
    main()
