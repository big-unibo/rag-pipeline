from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar


Item = TypeVar("Item")
Result = TypeVar("Result")


def execute_batched(
    items: Sequence[Item],
    process: Callable[[Item, int, int], Result],
    question_batch_size: int,
    on_question_batch: Callable[[list[Result]], None] | None = None,
) -> list[Result]:
    """Process questions uniformly with checkpointed batches.

    A pending batch is flushed before a processing exception is re-raised, so
    completed work is available to the caller's temporary output session.
    """
    if not isinstance(question_batch_size, int) or question_batch_size <= 0:
        raise ValueError("question_batch_size must be a positive integer")

    total = len(items)
    results: list[Result] = []
    pending: list[Result] = []
    for position, item in enumerate(items, start=1):
        try:
            result = process(item, position, total)
        except Exception:
            if pending and on_question_batch:
                on_question_batch(pending)
            raise
        results.append(result)
        pending.append(result)
        if len(pending) >= question_batch_size:
            if on_question_batch:
                on_question_batch(pending)
            pending = []

    if pending and on_question_batch:
        on_question_batch(pending)
    return results
