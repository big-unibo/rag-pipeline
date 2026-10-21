from __future__ import annotations

import json
import string

import numpy as np

from rag_pipeline.models import Answer, Question
from rag_pipeline.ollama import OllamaClient

from .interface import Evaluator


class BasicEvaluator(Evaluator):
    """Run all standard answer-quality metrics as one evaluator."""

    def __init__(self, config: dict):
        from rag_pipeline.stages.factory import build_stage

        self.ollama = OllamaClient(config)
        self.temperature = float(config.get("temperature", 0))
        self.embedder = build_stage(config.get("embedding", {}), "OllamaEmbedder")

    def evaluate(self, question: Question, answer: Answer) -> dict:
        ground_truth = question.ground_truth
        answer_text = answer.text.strip()
        metrics = {
            "answered": bool(answer_text),
            "has_ground_truth": ground_truth is not None,
        }
        if ground_truth:
            normalized_answer = answer_text.casefold()
            normalized_truth = ground_truth.strip().casefold()
            translator = str.maketrans("", "", string.punctuation)
            cleaned_answer = normalized_answer.translate(translator)
            cleaned_truth = normalized_truth.translate(translator)
            truth_terms = [term for term in cleaned_truth.split() if len(term) > 3]
            matches = sum(1 for term in truth_terms if term in cleaned_answer)
            metrics["exact_match"] = normalized_answer == normalized_truth
            metrics["ground_truth_term_recall"] = matches / len(truth_terms) if truth_terms else 0

            answer_vector, truth_vector = self.embedder.embed_texts(
                [answer_text, ground_truth]
            )
            metrics["cosine_similarity"] = cosine_similarity(answer_vector, truth_vector)
        metrics.update(self._evaluate_with_llm_judge(question, answer))
        return metrics

    def _evaluate_with_llm_judge(self, question: Question, answer: Answer) -> dict:
        if not question.ground_truth:
            return {"llm_judge_available": False}

        prompt = (
            "Evaluate the answer against the ground truth.\n"
            "Return only JSON with keys score and rationale.\n"
            "score must be a number from 0 to 1.\n\n"
            f"Question: {question.text}\n"
            f"Ground truth: {question.ground_truth}\n"
            f"Answer: {answer.text}\n"
        )
        body = self.ollama.generate(
            prompt,
            temperature=self.temperature,
            response_format="json",
        )
        raw_text = body.get("response", "")
        try:
            judged = json.loads(raw_text)
        except json.JSONDecodeError:
            return {"llm_judge_score": 0.0, "llm_judge_rationale": raw_text}
        return {
            "llm_judge_score": float(judged.get("score", 0)),
            "llm_judge_rationale": judged.get("rationale", ""),
        }


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm == 0 or second_norm == 0:
        return 0.0
    return float(np.dot(first, second) / (first_norm * second_norm))
