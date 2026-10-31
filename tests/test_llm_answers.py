"""The cited-answer schema, the numbered-passage prompt and the validating parser."""

from __future__ import annotations

import json
from typing import Any

import pytest

pytest.importorskip("evalharness")

from evalharness.llm.base import Completion  # noqa: E402

from rag_qa_eval.llm_answers import AnswerFormatError, CitedAnswer, build_messages, parse_answer  # noqa: E402

PASSAGES = [f"Passage text number {i}." for i in range(1, 6)]


def completion(payload: dict[str, Any] | str | None, *, as_data: bool = True) -> Completion:
    """A model reply: structured `data`, or only JSON text, or raw text."""
    text = payload if isinstance(payload, str) else json.dumps(payload)
    data = payload if (as_data and isinstance(payload, dict)) else None
    return Completion(
        text=text or "",
        data=data,
        input_tokens=10,
        output_tokens=5,
        latency_s=0.1,
        model="m",
        billing="local",
    )


def test_prompt_numbers_passages_in_rank_order() -> None:
    messages = build_messages("Who founded Rome?", PASSAGES)
    user = messages[-1].content
    positions = [user.index(f"[{i}] Passage text number {i}.") for i in range(1, 6)]
    assert positions == sorted(positions)
    assert user.rstrip().endswith("Who founded Rome?")
    assert messages[0].role == "system" and "abstain" in messages[0].content


def test_passages_with_bracket_numbers_are_kept_verbatim() -> None:
    passages = ["See also [3] and [4] in the text.", "Another passage."]
    user = build_messages("q?", passages)[-1].content
    assert "[1] See also [3] and [4] in the text." in user and "[2] Another passage." in user


def test_prompt_contains_no_gold_answer() -> None:
    gold = "Romulus the founder"
    messages = build_messages("Who founded Rome?", PASSAGES)  # the gold answer is not in the passages
    assert all(gold not in m.content for m in messages)


def test_valid_answer_parses() -> None:
    parsed = parse_answer(completion({"answer": "Romulus", "citations": [2], "abstain": False}), 5)
    assert parsed == CitedAnswer(answer="Romulus", citations=[2], abstain=False)


def test_json_in_text_is_accepted_when_data_is_missing() -> None:
    reply = completion('```json\n{"answer": "Romulus", "citations": [1], "abstain": false}\n```')
    assert parse_answer(reply, 5).citations == [1]


@pytest.mark.parametrize("bad", [0, 6, -1])
def test_citation_out_of_range_rejected(bad: int) -> None:
    with pytest.raises(AnswerFormatError, match=str(bad)):
        parse_answer(completion({"answer": "x", "citations": [bad], "abstain": False}), 5)


def test_zero_based_index_is_rejected() -> None:
    # a model that counts from zero cites 0 to 4; 0 must not be silently accepted
    with pytest.raises(AnswerFormatError, match="0"):
        parse_answer(completion({"answer": "x", "citations": [0, 1], "abstain": False}), 5)


def test_duplicate_citations_deduplicated() -> None:
    parsed = parse_answer(completion({"answer": "x", "citations": [2, 2, 1], "abstain": False}), 5)
    assert parsed.citations == [2, 1]


def test_abstain_with_nonempty_answer_rejected() -> None:
    with pytest.raises(AnswerFormatError, match="abstain"):
        parse_answer(completion({"answer": "Romulus", "citations": [1], "abstain": True}), 5)


def test_empty_answer_without_abstain_rejected() -> None:
    with pytest.raises(AnswerFormatError, match="abstain"):
        parse_answer(completion({"answer": "  ", "citations": [], "abstain": False}), 5)


def test_answer_with_no_citation_rejected() -> None:
    with pytest.raises(AnswerFormatError, match="citation"):
        parse_answer(completion({"answer": "Romulus", "citations": [], "abstain": False}), 5)


def test_abstain_is_valid_and_clears_citations() -> None:
    parsed = parse_answer(completion({"answer": "", "citations": [3], "abstain": True}), 5)
    assert parsed.abstain and parsed.answer == "" and parsed.citations == []


@pytest.mark.parametrize(
    "reply",
    [
        completion("I cannot decide."),
        completion({"answer": "x", "abstain": False}),  # missing citations
        completion({"answer": 7, "citations": [1], "abstain": False}),  # wrong type
        completion({"answer": "x", "citations": ["one"], "abstain": False}),
    ],
)
def test_malformed_replies_raise_answer_format_error(reply: Completion) -> None:
    with pytest.raises(AnswerFormatError):
        parse_answer(reply, 5)


def test_system_prompt_limits_abstention_to_passages_without_the_answer() -> None:
    """Chosen on the calibration split (docs/llm-answers.md): a prompt that stresses abstention made a small
    model abstain on every question, so the rule must be 'only when no passage contains the answer'."""
    from rag_qa_eval.llm_answers import SYSTEM_PROMPT

    assert "only when none of the passages contains the answer" in SYSTEM_PROMPT
    assert "Never answer from memory" not in SYSTEM_PROMPT


@pytest.mark.parametrize("word", ["abstain", "Abstain.", "none", "N/A", "no answer", "unanswerable"])
def test_abstain_flag_with_a_placeholder_word_and_no_citation_is_an_abstention(word: str) -> None:
    """A small model wrote the word 'abstain' into `answer` (930 such replies in the first full run): the flag
    and the empty citation list say what it meant, so this is an abstention, not an invalid reply."""
    parsed = parse_answer(completion({"answer": word, "citations": [], "abstain": True}), 5)
    assert parsed == CitedAnswer(answer="", citations=[], abstain=True)


def test_abstain_flag_with_a_real_answer_stays_invalid() -> None:
    with pytest.raises(AnswerFormatError, match="abstain"):
        parse_answer(completion({"answer": "Romulus", "citations": [], "abstain": True}), 5)
