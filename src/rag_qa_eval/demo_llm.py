"""Precomputed LLM answers for the browser demo's examples (needs the optional `llm` extra).

The demo never calls a model when a visitor clicks an example: the answer is produced here, ahead of time,
from the same prompt and the same top-5 passages as the evaluation, and read back from the response cache
that the `llm` stage filled.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from evalharness.llm.base import LLMClient

from .llm_answers import generate_answer


def llm_demo_block(
    question: str, passages: Sequence[Mapping[str, Any]], *, client: LLMClient, model: str
) -> dict[str, Any]:
    """One example's LLM answer, with the numbered passages it saw, as plain JSON-ready data."""
    result = generate_answer(question, [str(p["text"]) for p in passages], client=client, model=model)
    return {
        "model": model,
        "answer": result.answer or "",
        "citations": list(result.citations),
        "abstained": result.abstained,
        "valid": result.valid,
        "passages": [dict(p) for p in passages],
    }
