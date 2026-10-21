from __future__ import annotations

import re
from typing import Any

from rag_pipeline.models import Answer, Question
from rag_pipeline.ollama import OllamaClient

from .interface import Evaluator


VRAG_RUBRIC = """
1: The response is not aligned with the ground truth or is off-topic; includes hallucination.
2: The response admits it cannot provide an answer or lacks context; honest.
3: The response is relevant but contains notable discrepancies or inaccuracies.
4: The response is acceptable, sufficient but not exhaustive.
5: The response is fully accurate and comprehensive, based on the ground truth.
""".strip()

VRAG_PROMPT = """You are an AI assistant. In the following task, you are given a Question, a RAG application's response, and a Ground-truth Answer referred to as 'Label'.
Assess how well the RAG application's response aligns with the ground truth, using the grading rubric below:

[Start of Grading Rubric]
{rubric}
[End of Grading Rubric]

Treat the Label as the definitive answer.
Present your final score in the format: "[[score]]", followed by your justification.
Example: Score: [[3]], Reason: [[The RAG's response partially aligns with the ground truth but with some discrepancies]].
[Start of User Question]
{question_text}
[End of User Question]
[Start of Ground-truth Answer]
{ground_truth}
[End of Ground-truth Answer]
[Start of RAG's Application Response]
{rag_response}
[End of RAG's Application Response]
"""


class VragEvaluator(Evaluator):
    """Evaluate an answer with the five-level vRAG LLM-as-a-judge rubric."""

    def __init__(self, config: dict[str, Any]):
        self.ollama = OllamaClient(config)
        self.temperature = float(config.get("temperature", 0))
        self.rubric = config.get("rubric", VRAG_RUBRIC)

    def evaluate(self, question: Question, answer: Answer) -> dict[str, Any]:
        if not question.ground_truth:
            return {"vrag_score_available": False}

        prompt = VRAG_PROMPT.format(
            rubric=self.rubric,
            question_text=question.text,
            ground_truth=question.ground_truth,
            rag_response=answer.text,
        )
        raw_response = self._query(prompt)
        parsed = self.parse_judge_output(raw_response)
        return {
            "vrag_score_available": parsed["score"] is not None,
            "vrag_score": parsed["score"],
            "vrag_justification": parsed["justification"],
            "vrag_raw_response": parsed["raw_response"],
        }

    def _query(self, prompt: str) -> str:
        body = self.ollama.generate(
            prompt,
            temperature=self.temperature,
        )
        return str(body.get("response", ""))

    @staticmethod
    def parse_judge_output(raw_text: str) -> dict[str, Any]:
        score = None
        for candidate in re.findall(r"\[\[\s*([1-5])\s*\]\]", raw_text):
            score = int(candidate)
            break
        return {
            "score": score,
            "justification": raw_text.strip(),
            "raw_response": raw_text,
        }
