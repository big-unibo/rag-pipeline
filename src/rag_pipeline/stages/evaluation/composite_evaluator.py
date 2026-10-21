from rag_pipeline.models import Answer, Question

from .interface import Evaluator


class CompositeEvaluator(Evaluator):
    """Combine any configured evaluator implementations."""

    def __init__(self, config: dict):
        from rag_pipeline.stages.factory import build_stage

        self.evaluators = [
            build_stage(
                metric,
                "BasicEvaluator"
            )
            for metric in config.get("metrics", [{"strategy": "BasicEvaluator"}])
        ]

    def evaluate(self, question: Question, answer: Answer) -> dict:
        metrics = {}
        for evaluator in self.evaluators:
            metrics.update(evaluator.evaluate(question, answer))
        return metrics
