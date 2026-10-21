from __future__ import annotations

from rag_pipeline.models import Prompt, Question, RetrievedChunk
from rag_pipeline.stages.prompting.interface import PromptBuilder


PROMPT_TEMPLATES = {
    "strict_cited": """
You are an academic research assistant.
- Answer the question using *only* the provided context.
- If the answer is not present in the context, say: "I cannot answer based on the provided documents."
- In the answer, cite supporting chunks inline using [1], [2], etc.

Context:
{context}

Question: {question}

Answer:""",
}


class BasicPromptBuilder(PromptBuilder):
    def __init__(self, config: dict):
        self.template_name = config.get("template", "strict_cited")
        # custom_template lets a YAML config try a one-off prompt without
        # adding a permanent entry to PROMPT_TEMPLATES.
        self.template = config.get("custom_template") or PROMPT_TEMPLATES[self.template_name]

    def build(self, question: Question, chunks: list[RetrievedChunk]) -> Prompt:
        context = format_context(chunks)
        return Prompt(
            text=self.template.format(context=context, question=question.text),
            context=context,
            question=question,
            references=chunks,
        )


def format_context(chunks: list[RetrievedChunk]) -> str:
    # The numbered REF ID format is what the strict prompt asks the LLM to cite.
    # It also makes retrieved_chunks.csv easy to line up with an answer.
    parts: list[str] = []
    for index, item in enumerate(chunks, start=1):
        metadata = item.chunk.metadata
        source = metadata.get("title", item.chunk.document_id)
        source_path = metadata.get("source_path")
        source_type = metadata.get("source_type", "retrieval")
        chunk_id = metadata.get("chunk_id", item.chunk.id)
        path_text = f"; Source Path={source_path}" if source_path else ""
        parts.append(
            f"--- REF ID: [{index}] ---\n"
            f"Metadata: Source={source}; Source Type={source_type}; "
            f"Document ID={item.chunk.document_id}; Chunk ID={chunk_id}{path_text}; similarity={item.similarity}; Score={item.score}\n"
            f"Content: {item.chunk.text}\n"
        )
    return "\n".join(parts)
