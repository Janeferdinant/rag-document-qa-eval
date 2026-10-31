"""LLM answers with citations: the answer schema, the numbered-passage prompt and the validating parser.

Needs the optional `llm` extra (`pip install ".[llm]"`), which provides the model clients, response cache and
statistics from `llm-eval-harness`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from evalharness.llm.base import Completion, LLMClient, Message, extract_json
from pydantic import BaseModel, ValidationError

SYSTEM_PROMPT = (
    "Answer the question using the numbered passages. Copy the shortest span from a passage that fully "
    "answers it and cite the number of every passage you used (1-based, as shown). Set `abstain` to true, "
    "with an empty `answer` and no citations, only when none of the passages contains the answer."
)


# Words a model may write into `answer` instead of leaving it empty when it sets `abstain`: the flag says what it meant.
ABSTAIN_WORDS = frozenset({"abstain", "abstained", "none", "n/a", "no answer", "unanswerable"})


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
    placeholder = parsed.answer.strip().strip(".!").lower() in ABSTAIN_WORDS
    empty = not parsed.answer.strip() or (parsed.abstain and placeholder)
    if parsed.abstain != empty:
        raise AnswerFormatError("`abstain` must be true exactly when `answer` is empty")
    if parsed.abstain:
        return CitedAnswer(answer="", citations=[], abstain=True)
    citations = list(dict.fromkeys(parsed.citations))
    if not citations:
        raise AnswerFormatError("an answer needs at least one citation")
    return CitedAnswer(answer=parsed.answer.strip(), citations=citations, abstain=False)


@dataclass
class AnswerResult:
    """One question's outcome. `valid` is False when the model never produced a valid cited answer (after one
    retry); such a question is still counted, as a wrong answer, never dropped. `answer` is None when the model
    abstained or the reply was invalid (`abstained` tells which)."""

    answer: str | None
    citations: list[int]
    abstained: bool
    valid: bool
    invalid_reason: str | None = None
    completions: list[Completion] = field(default_factory=list)


def generate_answer(question: str, passages: Sequence[str], *, client: LLMClient, model: str) -> AnswerResult:
    """Ask `client` for a cited answer. An invalid reply is retried once with the validation error appended.
    No passages means abstain without a model call. A usage limit is not caught: the caller stops and resumes."""
    if not passages:
        return AnswerResult(answer=None, citations=[], abstained=True, valid=True)
    messages = build_messages(question, passages)
    completions: list[Completion] = []
    reason = ""
    for _ in range(2):
        try:
            c = client.complete(
                messages, model=model, temperature=0.0, seed=0, max_tokens=1024, response_schema=CitedAnswer
            )
        except ValueError as e:  # the client could not produce the schema (it validates and retries itself)
            reason = str(e)
            messages = [*messages, Message(role="user", content=_retry_text(reason))]
            continue
        completions.append(c)
        try:
            parsed = parse_answer(c, len(passages))
        except AnswerFormatError as e:
            reason = str(e)
            messages = [
                *messages,
                Message(role="assistant", content=c.text),
                Message(role="user", content=_retry_text(reason)),
            ]
            continue
        if parsed.abstain:
            return AnswerResult(None, [], abstained=True, valid=True, completions=completions)
        return AnswerResult(
            parsed.answer, parsed.citations, abstained=False, valid=True, completions=completions
        )
    return AnswerResult(
        None, [], abstained=False, valid=False, invalid_reason=reason, completions=completions
    )


def _retry_text(reason: str) -> str:
    return f"Your reply was not valid: {reason}. Reply again with a valid JSON object."
