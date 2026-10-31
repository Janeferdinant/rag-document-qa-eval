"""Shared fakes for the LLM-stage tests (needs the `llm` extra): a scripted answerer and a fake run."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from evalharness.llm.base import Completion, Message, UsageLimitError

from fakes import components, settings
from rag_qa_eval.eval_llm import ArmSpec, run_llm


class FakeAnswerer:
    """Answers 'Who founded TopicN?' from a passage that mentions the topic and 'by PersonN'; abstains otherwise.
    `mode` can make it return junk, or raise a usage limit at the n-th call."""

    def __init__(self, provider: str = "fake", mode: str = "ok", limit_at: int | None = None) -> None:
        self.provider, self.mode, self.limit_at, self.calls = provider, mode, limit_at, 0

    def complete(self, messages: list[Message], **kw: Any) -> Completion:
        self.calls += 1
        if self.limit_at is not None and self.calls == self.limit_at:
            raise UsageLimitError(self.provider)
        user = messages[-1].content
        if "Passages:" not in user:  # a judge call
            payload: dict[str, Any] = {"score": 2, "reasoning": "supported"}
        elif self.mode == "junk":
            return self._reply("not json at all", None, kw)
        else:
            question = re.search(r"^Question: (.*)$", user, flags=re.M)
            topic = re.search(r"(Topic\w+)", question.group(1)) if question else None
            payload = {"answer": "", "citations": [], "abstain": True}
            if topic and question and "founded" in question.group(1):
                for m in re.finditer(r"^\[(\d+)\] (.*)$", user, flags=re.M):
                    p = re.search(r"by (Person\w+)", m.group(2))
                    if p and topic.group(1) in m.group(2):
                        payload = {"answer": p.group(1), "citations": [int(m.group(1))], "abstain": False}
                        break
        return self._reply(json.dumps(payload), payload, kw)

    def _reply(self, text: str, data: dict[str, Any] | None, kw: dict[str, Any]) -> Completion:
        return Completion(
            text=text, data=data, input_tokens=100, output_tokens=10, latency_s=0.25, model=kw["model"],
            model_digest="digest-1", cost_usd=0.001, billing="subscription",
        )  # fmt: skip


ARMS = (
    ArmSpec("fake-local", "fake-model", "ollama", "all"),
    ArmSpec("fake-hosted", "fake-model", "claude_code", "sample"),
)


def n_test(world: dict[str, Any]) -> int:
    return len(world["prep"].sample("test", settings().n_test_questions, settings().seed, False))


def run_fake(world: dict[str, Any], tmp: Path, factory: Any = None, **kw: Any) -> dict[str, Any]:
    msgs: list[str] = []
    return run_llm(
        world["prep"], settings(), components(), tmp / "cache", world["retrieval"], world["reader"], ARMS,
        factory or (lambda arm: FakeAnswerer(arm.provider)), sample_n=20, log=msgs.append,
        judge=(FakeAnswerer("judge"), "qwen3:8b"), **kw,
    )  # fmt: skip
