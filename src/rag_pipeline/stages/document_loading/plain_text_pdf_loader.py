from __future__ import annotations

from pathlib import Path

from rag_pipeline.models import Document
from rag_pipeline.settings import resolve_path
from rag_pipeline.stages.document_loading.interface import DocumentLoader


class PlainTextPdfLoader(DocumentLoader):
    """Loads PDF files from a directory as documents."""

    def __init__(self, config: dict):
        self.input_dir = resolve_path(config.get("input_dir", "resources/datasets/movies/documents"))
        self.glob = config.get("glob", "*.pdf")

    def load_documents(self) -> list[Document]:
        try:
            import pdfplumber
        except ImportError as exc:
            raise RuntimeError("Install pdfplumber to use the local PDF loader.") from exc

        documents: list[Document] = []
        for path in self._document_paths():
            pages: list[str] = []
            print(f"Loading PDF: {path}")
            if path.suffix.lower() == ".pdf":
                with pdfplumber.open(path) as pdf:
                    for page_number, page in enumerate(pdf.pages, start=1):
                        text = page.extract_text() or ""
                        if text.strip():
                            pages.append(f"\n\n[Page {page_number}]\n{text}")
                documents.append(
                    Document(
                        id=path.stem,
                        title=path.name,
                        text="".join(pages).strip(),
                        metadata={"source_path": str(path)},
                    )
                )
        return documents

    def _document_paths(self) -> list[Path]:
        # pathlib's "*.pdf" glob is case-sensitive on some systems; this branch
        # keeps ".PDF" files working in Linux CI.
        if self.glob == "*.pdf":
            return sorted(path for path in self.input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pdf")
        return sorted(self.input_dir.glob(self.glob))
