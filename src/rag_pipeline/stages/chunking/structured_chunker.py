
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
)

from rag_pipeline.models import Chunk, Document
from rag_pipeline.stages.chunking.interface import Chunker


class StructuredChunker(Chunker):
    """Splits documents into fixed-size chunks considering the Markdown structure."""
    def __init__(self, config: dict) -> None:
        self.chunk_size = int(config.get("chunk_size", 200))
        self.chunk_overlap = int(config.get("chunk_overlap", 0))
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ]
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=lambda text: len(text.split()),
            separators=["\n\n", "\n", " "],
        )

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []

        for document in documents:
            sections = self.header_splitter.split_text(document.text)
            chunk_index = 0

            for section in sections:
                texts = self.text_splitter.split_text(
                    section.page_content
                )

                for text in texts:
                    chunks.append(
                        Chunk(
                            id=f"{document.id}:{chunk_index}",
                            document_id=document.id,
                            text=text,
                            metadata={
                                **document.metadata,
                                "title": document.title,
                                "chunk_index": chunk_index,
                                **section.metadata,
                            },
                        )
                    )
                    chunk_index += 1

        return chunks