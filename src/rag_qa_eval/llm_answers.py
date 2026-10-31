"""LLM answers with citations: the answer schema, the numbered-passage prompt and the validating parser.

Needs the optional `llm` extra (`pip install ".[llm]"`), which provides the model clients, response cache and
statistics from `llm-eval-harness`.
"""

from __future__ import annotations

from collections.abc import Sequence

from evalharness.llm.base import Completion, Message, extract_json
from pydantic import BaseModel, ValidationError

SYSTEM_PROMPT = (
    "You answer questions using only the numbered passages you are given. "
    "Reply with the shortest span that fully answers the question, copied from a passage, "
    "and cite the number of every passage the answer relies on (the numbers are 1-based, as shown). "
    "If the passages do not contain the answer, abstain: set `abstain` to true, `answer` to an empty "
    "string and `citations` to an empty list. Never answer from memory."
)


class CitedAnswer(BaseModel):
    answer: str
    citations: list[int]
    abstain: bool


class AnswerFormatError(ValueError):
    """The model's reply is not a valid cited answer. The message says what is wrong, for the retry."""


def build_messages(question: str, passages: Sequence[str]) -> list[Message]:
    """The system rules and the user turn: passages numbered [1]..[n] in the given (rank) order, then the question.

    Passage text is kept as it is (whitespace collapsed), even when it contains bracketed numbers of its own."""
    listing = "\n".join(f"[{i}] {' '.join(p.split())}" for i, p in enumerate(passages, start=1))
    user = f"Passages:\n{listing}\n\nQuestion: {question}"
    return [Message(role="system", content=SYSTEM_PROMPT), Message(role="user", content=user)]


def parse_answer(c: Completion, n_passages: int) -> CitedAnswer:
    """Validate a model reply: structured data (or JSON in the text), citations in 1..n, no duplicates,
    `abstain` true exactly when the answer is empty, and at least one citation for an answer."""
    try:
        payload = c.data if c.data is not None else extract_json(c.text)
    except ValueError as e:
        raise AnswerFormatError(f"the reply is not a JSON object: {e}") from e
    try:
        parsed = CitedAnswer.model_validate(payload)
    except ValidationError as e:
        first = e.errors()[0]
        where = ".".join(str(p) for p in first["loc"]) or "(root)"
        raise AnswerFormatError(f"the reply does not match the schema at {where}: {first['msg']}") from e
    bad = [n for n in parsed.citations if not 1 <= n <= n_passages]
    if bad:
        raise AnswerFormatError(f"citation {bad[0]} is outside 1..{n_passages}; passages are numbered from 1")
    empty = not parsed.answer.strip()
    if parsed.abstain != empty:
        raise AnswerFormatError("`abstain` must be true exactly when `answer` is empty")
    if parsed.abstain:
        return CitedAnswer(answer="", citations=[], abstain=True)
    citations = list(dict.fromkeys(parsed.citations))
    if not citations:
        raise AnswerFormatError("an answer needs at least one citation")
    return CitedAnswer(answer=parsed.answer.strip(), citations=citations, abstain=False)
