from typing import Protocol

from rag_pipeline.models import Document


class DocumentLoader(Protocol):
    """Loads raw documents from a configured source."""

    def load_documents(self) -> list[Document]:
        ...