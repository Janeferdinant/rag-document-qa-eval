"""The answer generator (one retry, invalid answers counted) and the adapter over evalharness clients."""

from __future__ import annotations

import json
from typing import Any

import pytest

pytest.importorskip("evalharness")

from evalharness.llm.base import Completion, Message, UsageLimitError  # noqa: E402

from rag_qa_eval.llm import HarnessClient, LLMReader, MockLLMClient, build_prompt, parse_llm_answer  # noqa: E402
from rag_qa_eval.llm_answers import generate_answer  # noqa: E402

PASSAGES = ["Rome was founded by Romulus.", "The Tiber runs through Rome."]
VALID = {"answer": "Romulus", "citations": [1], "abstain": False}


class Scripted:
    """An evalharness-style client that replays replies in order and records what it was asked."""

    provider = "scripted"

    def __init__(self, *replies: dict[str, Any] | str | Exception) -> None:
        self.replies = list(replies)
        self.calls: list[list[Message]] = []
        self.kwargs: list[dict[str, Any]] = []

    def complete(self, messages: list[Message], **kw: Any) -> Completion:
        self.calls.append(list(messages))
        self.kwargs.append(kw)
        reply = self.replies[min(len(self.calls), len(self.replies)) - 1]
        if isinstance(reply, Exception):
            raise reply
        text = reply if isinstance(reply, str) else json.dumps(reply)
        data = reply if isinstance(reply, dict) else None
        return Completion(
            text=text,
            data=data,
            input_tokens=100,
            output_tokens=10,
            latency_s=0.5,
            model=kw["model"],
            billing="local",
        )


def test_answer_end_to_end_mock() -> None:
    client = Scripted(VALID)
    result = generate_answer("Who founded Rome?", PASSAGES, client=client, model="m")
    assert (result.answer, result.citations, result.abstained, result.valid) == ("Romulus", [1], False, True)
    assert len(result.completions) == 1 and result.invalid_reason is None
    assert client.kwargs[0]["temperature"] == 0.0 and client.kwargs[0]["seed"] == 0
    assert client.kwargs[0]["response_schema"].__name__ == "CitedAnswer"


def test_abstention_is_a_valid_result() -> None:
    result = generate_answer(
        "Who painted Rome?",
        PASSAGES,
        client=Scripted({"answer": "", "citations": [], "abstain": True}),
        model="m",
    )
    assert result.abstained and result.answer is None and result.valid


def test_invalid_output_retried_once() -> None:
    client = Scripted({"answer": "Romulus", "citations": [9], "abstain": False}, VALID)
    result = generate_answer("Who founded Rome?", PASSAGES, client=client, model="m")
    assert (
        result.valid
        and result.answer == "Romulus"
        and len(client.calls) == 2
        and len(result.completions) == 2
    )
    retry = client.calls[1]
    assert retry[-1].role == "user" and "citation 9" in retry[-1].content
    assert any(m.role == "assistant" for m in retry)  # the model sees what it said


def test_invalid_after_retry_counted() -> None:
    bad = {"answer": "Romulus", "citations": [], "abstain": False}
    client = Scripted(bad, bad)
    result = generate_answer("Who founded Rome?", PASSAGES, client=client, model="m")
    assert not result.valid and result.invalid_reason and "citation" in result.invalid_reason
    assert result.answer is None and not result.abstained and len(client.calls) == 2


def test_client_that_cannot_produce_the_schema_is_counted_invalid_not_raised() -> None:
    client = Scripted(ValueError("reply does not match response_schema CitedAnswer"))
    result = generate_answer("q?", PASSAGES, client=client, model="m")
    assert not result.valid and "response_schema" in (result.invalid_reason or "") and len(client.calls) == 2


def test_no_passages_abstains_without_calling_the_model() -> None:
    client = Scripted(VALID)
    result = generate_answer("Who founded Rome?", [], client=client, model="m")
    assert result.abstained and result.valid and client.calls == [] and result.completions == []


def test_usage_limit_error_propagates() -> None:
    with pytest.raises(UsageLimitError):
        generate_answer("q?", PASSAGES, client=Scripted(UsageLimitError("claude_code")), model="m")


# ---- the adapter and the unchanged old interface ------------------------------------------------


def test_existing_llm_interface_unchanged() -> None:
    chunks = ["Normandy is a region in France.", "Paris is the capital."]
    answer = LLMReader(MockLLMClient()).answer("Where is Normandy?", chunks)
    assert answer.text is not None and "Normandy" in answer.text and answer.chunk_index == 0
    prompt = build_prompt("Where is Normandy?", chunks)
    assert "[1] Normandy is a region in France." in prompt and prompt.endswith("Answer:")
    assert parse_llm_answer("Normandy [2]").citations == [2]
    assert parse_llm_answer("UNANSWERABLE").text is None


def test_harness_client_wraps_an_evalharness_client() -> None:
    stub = Scripted("Normandy [1]")
    client = HarnessClient(stub, model="qwen3:8b")
    assert client.name == "scripted:qwen3:8b"
    assert client.generate("a prompt", temperature=0.0, max_output_tokens=64) == "Normandy [1]"
    assert stub.calls[0][-1].content == "a prompt" and stub.kwargs[0]["max_tokens"] == 64
    answer = LLMReader(client).answer("Where is Normandy?", ["Normandy is a region in France."])
    assert answer.text == "Normandy" and answer.chunk_index == 0
