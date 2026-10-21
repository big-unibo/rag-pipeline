from __future__ import annotations

import re
from typing import Any
from rapidfuzz import fuzz

from rag_pipeline.models import Answer, Question, RetrievedChunk
from rag_pipeline.ollama import OllamaClient

from .interface import Evaluator


SUPPORT_PROMPT = """You are checking whether a claim is really supported by a given set of citations.

Use this logic:
- The Citation Texts are the evidence you must use for the decision.
- The Claim is the statement that must be verified.
- Use only the Citation Texts. Do not use outside knowledge.
- Mark the Claim as supported only if the Citation Texts contain enough evidence to verify the full Claim.
- If the Citation Texts support only part of the Claim, mark it as unsupported.

Evaluate two things:
1. Joint Support: do all Citation Texts together support the full Claim?
2. Supported Citations: which individual Citation Texts support the full Claim on their own?

Return exactly this format:
Joint Support: [[SUPPORTED]] or [[UNSUPPORTED]]
Supported Citations: [[1, 3]] or [[]]

[Start of User Question]
{question_text}
[End of User Question]

[Start of Claim]
{claim_text}
[End of Claim]

[Start of Citation Texts]
{citation_texts}
[End of Citation Texts]
"""

DEFAULT_REFUSAL_FLAG = "I apologize, but I couldn't find an answer."
DEFAULT_REFUSAL_THRESHOLD = 85


class TrustScoreEvaluator(Evaluator):
    """Compute grounded refusal, answer correctness, and citation scores with Ollama."""

    def __init__(self, config: dict[str, Any]):
        self.ollama = OllamaClient(config)
        self.temperature = float(config.get("temperature", 0))
        self.refusal_flag = config.get("refusal_flag", DEFAULT_REFUSAL_FLAG)
        self.refusal_threshold = float(config.get("refusal_threshold", DEFAULT_REFUSAL_THRESHOLD))

    def evaluate(self, question: Question, answer: Answer) -> dict[str, Any]:
        if not question.ground_truth:
            return {"trust_score_available": False}

        claims = split_claims(question.ground_truth)
        references = answer.prompt.references
        answers_found = []
        for reference in references:
            evidence = format_reference(reference)
            answers_found.append([
                self._claim_is_supported(question.text, claim, evidence)
                for claim in claims
            ])

        supported = get_supported_claims(answers_found, len(claims))
        answerable = any(supported)
        refused = is_refusal_response(answer.text, self.refusal_flag, self.refusal_threshold)
        answered = not refused
        f1_gr = 100.0 if (answerable and answered) or (not answerable and refused) else 0.0

        claim_matches = [
            self._claim_is_supported(question.text, claim, answer.text)
            for claim in claims
        ]
        supported_count = sum(supported)
        supported_coverage = (
            100 * sum(match for match, is_supported in zip(claim_matches, supported) if is_supported) / supported_count
            if supported_count else 0.0
        )
        calibrated_answered = supported_coverage if answered and answerable else (0.0 if answered else None)
        calibrated_answerable = supported_coverage if answerable and answered else (0.0 if answerable else None)
        f1_ac = average_available([calibrated_answered, calibrated_answerable])

        grounding = self._evaluate_citations(question.text, answer.text, references)
        trust_score = calculate_trust_score(answerable, answered, f1_ac, grounding["f1_gc"])
        return {
            "trust_score_available": True,
            "is_answerable": answerable,
            "answered": answered,
            "refused": refused,
            "supported_claims": supported,
            "claim_match_scores": claim_matches,
            "supported_coverage": supported_coverage,
            "f1_gr": f1_gr,
            "f1_ac": f1_ac,
            **grounding,
            "trust_score": trust_score,
        }

    def _claim_is_supported(self, question: str, claim: str, evidence: str) -> int:
        if not evidence.strip():
            return 0
        raw = self._query(
            SUPPORT_PROMPT.format(
                question_text=question,
                claim_text=claim,
                citation_texts=evidence,
            )
        )
        return int(parse_joint_support(raw))

    def _query(self, prompt: str) -> str:
        body = self.ollama.generate(
            prompt,
            temperature=self.temperature,
        )
        return str(body.get("response", ""))

    def _evaluate_citations(
        self,
        question: str,
        response: str,
        references: list[RetrievedChunk],
    ) -> dict[str, Any]:
        documents = {str(reference.chunk.id): reference.chunk.text for reference in references}
        total_claims = total_citations = supported_claims = supported_citations = 0
        details = []
        for sentence in split_sentences(response):
            citations = get_citations(sentence)
            if not citations:
                continue
            total_claims += 1
            total_citations += len(citations)
            citation_texts = "\n\n".join(
                f"[{citation}]\n{documents.get(citation, '')}" for citation in citations
            )
            raw = self._query(
                SUPPORT_PROMPT.format(
                    question_text=question,
                    claim_text=remove_citations(sentence),
                    citation_texts=citation_texts,
                )
            )
            joint = parse_joint_support(raw)
            individual = parse_supported_citations(raw)
            supported_claims += int(joint)
            supported_citations += sum(citation in individual for citation in citations)
            details.append({
                "sentence": sentence,
                "citations": citations,
                "joint_supported": joint,
                "individually_supported_citations": individual,
            })

        precision = 100 * safe_divide(supported_citations, total_citations) if safe_divide(supported_citations, total_citations) is not None else None
        recall = 100 * safe_divide(supported_claims, total_claims) if safe_divide(supported_claims, total_claims) is not None else None
        return {
            "citation_precision": precision,
            "citation_recall": recall,
            "f1_gc": calculate_f1(precision, recall),
            "grounding_details": details,
        }


def normalize_answer(text: str) -> str:
    text = re.sub(r"\b(a|an|the)\b", " ", text.lower())
    return " ".join(re.sub(r"[^\w\s]", "", text).split())


def is_refusal_response(response: str, refusal_flag: str, threshold: float) -> bool:
    response = normalize_answer(response)
    refusal = normalize_answer(refusal_flag)
    return fuzz.partial_ratio(refusal, response) > threshold


def split_sentences(text: str) -> list[str]:
    return [part.strip(" -\t") for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip(" -\t")]


def split_claims(text: str) -> list[str]:
    return [remove_citations(part) for part in split_sentences(text)]


def remove_citations(text: str) -> str:
    return re.sub(r"\s*\[\d+\]", "", text).strip()


def format_reference(reference: RetrievedChunk) -> str:
    """Format one retrieved chunk as the citation ID and its text."""
    return f"[{reference.chunk.id}]\n{reference.chunk.text}"


def get_supported_claims(documents: list[list[int]], count: int) -> list[int]:
    supported = [0] * count
    for answers_found in documents:
        for index, value in enumerate(answers_found):
            if value:
                supported[index] = 1
    return supported


def get_citations(sentence: str) -> list[str]:
    citations = re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", sentence)
    return [n.strip() for citation in citations for n in citation.split(",")]


def parse_joint_support(raw: str) -> bool:
    match = re.search(r"Joint Support:\s*\[\[(SUPPORTED|UNSUPPORTED)\]\]", raw, re.IGNORECASE)
    return bool(match and match.group(1).upper() == "SUPPORTED")


def parse_supported_citations(raw: str) -> list[str]:
    match = re.search(r"Supported Citations:\s*\[\[([^\]]*)\]\]", raw, re.IGNORECASE)
    return [value.strip() for value in match.group(1).split(",") if value.strip().isdigit()] if match else []


def safe_divide(value: int, total: int) -> float | None:
    return value / total if total else None


def calculate_f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def average_available(values: list[float | None]) -> float | None:
    available = [value for value in values if value is not None]
    return sum(available) / len(available) if available else None


def calculate_trust_score(answerable: bool, answered: bool, f1_ac: float | None, f1_gc: float | None) -> float | None:
    if answerable and answered:
        return average_available([f1_ac, f1_gc])
    if not answerable and not answered:
        return None
    return 0.0
