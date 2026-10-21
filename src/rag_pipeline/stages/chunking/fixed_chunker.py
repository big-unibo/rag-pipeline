from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_pipeline.models import Chunk, Document
from rag_pipeline.stages.chunking.interface import Chunker


class FixedChunker(Chunker):
    """Splits documents into fixed-size, overlapping text chunks."""

    def __init__(self, config: dict):
        self.chunk_size = int(config.get("chunk_size", 1000))
        self.chunk_overlap = int(config.get("chunk_overlap", 200))

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks: list[Chunk] = []
        for document in documents:
            texts = splitter.split_text(f"Title: {document.title}\n\n{document.text}")
            for index, text in enumerate(texts):
                chunks.append(
                    Chunk(
                        id=f"{document.id}:{index}",
                        document_id=document.id,
                        text=text,
                        metadata={
                            "title": document.title,
                            "chunk_index": index,
                            **document.metadata,
                        },
                    )
                )
        return chunks
