from typing import Protocol
from rag_pipeline.models import Answer, Question


class Evaluator(Protocol):
    def evaluate(self, question: Question, answer: Answer) -> dict:
        ...
