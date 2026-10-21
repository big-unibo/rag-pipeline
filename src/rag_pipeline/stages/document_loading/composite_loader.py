from __future__ import annotations

from typing import Any

from rag_pipeline.models import Document
from rag_pipeline.stages.factory import build_stage
from rag_pipeline.stages.document_loading.interface import DocumentLoader


class CompositeLoader:
    """Loads and combines documents from multiple configured loaders."""

    def __init__(self, config: dict[str, Any]):
        loader_configs = config.get("loaders")
        if not isinstance(loader_configs, list) or not loader_configs:
            raise ValueError("CompositeLoader requires a non-empty 'loaders' list.")
        if not all(isinstance(loader_config, dict) for loader_config in loader_configs):
            raise ValueError("Every CompositeLoader entry must be a loader configuration mapping.")

        shared_config = {
            key: value
            for key, value in config.items()
            if key not in {"strategy", "loaders"}
        }
        self.loaders: list[DocumentLoader] = [
            build_stage({**shared_config, **loader_config}, "PlainTextPdfLoader")
            for loader_config in loader_configs
        ]

    def load_documents(self) -> list[Document]:
        return [
            document
            for loader in self.loaders
            for document in loader.load_documents()
        ]
