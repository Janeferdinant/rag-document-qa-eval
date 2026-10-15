"""Error analysis: separate retrieval failures from reading failures."""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from collections.abc import Sequence
from typing import Any

from .qa_metrics import score_prediction

CATEGORIES = (
    "correct",
    "retrieval_failure",
    "reading_failure_abstained",
    "reading_failure_wrong_span",
    "boundary_mismatch",
    "correct_abstention",
    "false_answer",
)

DESCRIPTIONS = {
    "correct": "answerable question, exact match with a gold answer",
    "retrieval_failure": "answerable, but no retrieved chunk contains a gold answer string",
    "reading_failure_abstained": "answerable, answer in the retrieved chunks, reader abstained",
    "reading_failure_wrong_span": "answerable, answer in the retrieved chunks, reader gave a "
    "different span (F1 < 0.5)",
    "boundary_mismatch": "answerable, reader's span overlaps the gold answer (0 < F1 < 1, not EM)",
    "correct_abstention": "unanswerable question, reader abstained",
    "false_answer": "unanswerable question, reader gave an answer",
}


def categorize(*, answerable: bool, abstained: bool, in_context: bool, em: float, f1: float) -> str:
    if not answerable:
        return "correct_abstention" if abstained else "false_answer"
    if em >= 1.0:
        return "correct"
    if not in_context:
        return "retrieval_failure"
    if abstained:
        return "reading_failure_abstained"
    return "boundary_mismatch" if f1 >= 0.5 else "reading_failure_wrong_span"


def category_counts(categories: Sequence[str]) -> dict[str, int]:
    unknown = set(categories) - set(CATEGORIES)
    if unknown:
        raise ValueError(f"unknown categories: {sorted(unknown)}")
    c = Counter(categories)
    return {name: c.get(name, 0) for name in CATEGORIES}


def question_metrics(prediction: str | None, golds: Sequence[str]) -> dict[str, float]:
    em, f1 = score_prediction(prediction, golds)
    return {"em": em, "f1": f1}


def anonymise_id(qid: str) -> str:
    return hashlib.sha256(qid.encode()).hexdigest()[:8]


def pick_examples(rows: Sequence[dict[str, Any]], per_category: int, seed: int) -> list[dict[str, Any]]:
    """Seeded pick of up to `per_category` rows from each category (order: CATEGORIES)."""
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for cat in CATEGORIES:
        pool = sorted((r for r in rows if r["category"] == cat), key=lambda r: r["qid"])
        out.extend(rng.sample(pool, min(per_category, len(pool))))
    return out
