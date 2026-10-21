from __future__ import annotations

import json

from rag_pipeline.models import Question, RetrievedChunk
from rag_pipeline.ollama import OllamaClient


class OllamaReranker:
    def __init__(self, config: dict):
        self.top_k = int(config.get("top_k", 5))
        self.ollama = OllamaClient(config)
        self.temperature = float(config.get("temperature", 0))

    def rank(self, question: Question, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []

        for item in chunks:
            item.score = self.score(question, item)

        ranked = sorted(chunks, key=lambda item: item.score or 0.0, reverse=True)
        for index, item in enumerate(ranked, start=1):
            item.rank = index
        return ranked[: self.top_k]

    def score(self, question: Question, chunk: RetrievedChunk) -> float:
        prompt = RERANK_PROMPT.format(question=question.text, chunk=chunk.chunk.text)
        body = self.ollama.generate(
            prompt,
            temperature=self.temperature,
            response_format="json",
        )
        text = body.get("response", "")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return 0.0
        return max(0.0, min(1.0, float(payload.get("score", 0.0))))


RERANK_PROMPT = """Score how relevant this context chunk is for answering the question.
Return only JSON with one key, score.
score must be a number from 0 to 1, where 1 means directly answers the question and 0 means irrelevant.

Question: {question}
Context chunk: {chunk}
"""
