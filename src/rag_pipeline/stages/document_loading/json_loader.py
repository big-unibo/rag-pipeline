from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag_pipeline.models import Document
from rag_pipeline.settings import resolve_path
from rag_pipeline.stages.document_loading.interface import DocumentLoader


class BasicJsonLoader(DocumentLoader):
    """Loads JSON documents with a text, context, or content field."""

    def __init__(self, config: dict):
        self.input_dir = resolve_path(config.get("input_dir", "resources/datasets/medical/documents"))
        self.glob = config.get("glob", "*.json")

    def load_documents(self) -> list[Document]:
        documents: list[Document] = []
        for path in self._document_paths():
            print(f"Loading JSON: {path}")
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            documents.extend(self._documents_from_payload(path, payload))
        return documents

    def _document_paths(self) -> list[Path]:
        return sorted(self.input_dir.glob(self.glob))

    def _documents_from_payload(self, path: Path, payload: Any) -> list[Document]:
        if isinstance(payload, dict):
            text = payload.get("text") or payload.get("context") or payload.get("content")
            if text is None:
                raise ValueError(f"JSON document {path} must contain text, context, or content.")
            return [self._document(path, str(text), payload)]

        if isinstance(payload, list):
            documents = []
            for index, item in enumerate(payload):
                if not isinstance(item, dict):
                    raise ValueError(f"JSON document {path} item {index} must be an object.")
                text = item.get("text") or item.get("context") or item.get("content")
                if text is None:
                    raise ValueError(f"JSON document {path} item {index} has no text field.")
                document_id = str(item.get("id", f"{path.stem}-{index}"))
                documents.append(self._document(path, str(text), item, document_id=document_id))
            return documents

        raise ValueError(f"JSON document {path} must contain an object or an array of objects.")

    @staticmethod
    def _document(
        path: Path,
        text: str,
        payload: dict[str, Any],
        document_id: str | None = None,
    ) -> Document:
        metadata = {
            key: value
            for key, value in payload.items()
            if key not in {"text", "context", "content"}
        }
        metadata["source_path"] = str(path)
        return Document(
            id=document_id or path.stem,
            title=str(payload.get("title") or payload.get("corpus_name") or path.name),
            text=text,
            metadata=metadata,
        )
