from __future__ import annotations

from pathlib import Path

from rag_pipeline.models import Document
from rag_pipeline.settings import resolve_path
from rag_pipeline.stages.document_loading.interface import DocumentLoader


class DoclingMarkdownPdfLoader(DocumentLoader):
    """Loads PDF files from a directory as Markdown documents."""

    def __init__(self, config: dict):
        self.input_dir = resolve_path(
            config.get(
                "input_dir",
                "resources/datasets/movies/documents",
            )
        )
        self.glob = config.get("glob", "*.pdf")

    def load_documents(self) -> list[Document]:
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as exc:
            raise RuntimeError(
                "Install pymupdf4llm to use the Markdown PDF loader."
            ) from exc

        documents: list[Document] = []

        for path in self._document_paths():
            print(f"Loading PDF as Markdown: {path}")

            converter = DocumentConverter()
            markdown = converter.convert(path).document.export_to_markdown()

            documents.append(
                Document(
                    id=path.stem,
                    title=path.name,
                    text=markdown.strip(),
                    metadata={
                        "source_path": str(path),
                        "format": "markdown",
                    },
                )
            )

        return documents

    def _document_paths(self) -> list[Path]:
        # Support uppercase extensions on Linux as well.
        if self.glob == "*.pdf":
            return sorted(
                path
                for path in self.input_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() == ".pdf"
            )

        return sorted(self.input_dir.glob(self.glob))