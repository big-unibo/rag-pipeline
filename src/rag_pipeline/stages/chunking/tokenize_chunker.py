from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from rag_pipeline.models import Chunk, Document
from rag_pipeline.stages.chunking.interface import Chunker


class TokenizeChunker(Chunker):
    """Splits documents into fixed token size chunk, overlapping of specified token number"""

    def __init__(self, config: dict):
        self.chunk_size = int(config.get("chunk_size", 256))
        self.chunk_overlap = int(config.get("chunk_overlap", 32))
        self.tokenizer = AutoTokenizer.from_pretrained(config.get("embed_model", "BAAI/bge-large-en-v1.5"))

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        index = 0
        for document in documents:
            tokens = self.tokenizer.encode(document.text, add_special_tokens=False)
            start = 0
            while start < len(tokens):
                end = min(start + self.chunk_size, len(tokens))
                chunk_tokens = tokens[start:end]
                chunk_text = self.tokenizer.decode(
                    chunk_tokens, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=True
                )
                chunks.append(
                    Chunk(
                        id=f"{document.id}:{index}",
                        document_id=document.id,
                        text=chunk_text,
                        metadata={
                            "title": document.title,
                            "chunk_index": index,
                            **document.metadata,
                        },
                    )
                )
                index +=1
                if end == len(tokens):
                    break
                start += self.chunk_size - self.chunk_overlap
        return chunks