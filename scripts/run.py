from __future__ import annotations

from rag_pipeline.runner import run_matrix
from rag_pipeline.settings import load_settings


def main() -> None:
    settings = load_settings("config/pipeline.yml")
    run_matrix(settings)


if __name__ == "__main__":
    main()
