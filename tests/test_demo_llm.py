"""Precomputed LLM answers for the browser demo (needs the `llm` extra): no models, no network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("evalharness")

from llm_fakes import FakeAnswerer  # noqa: E402
from rag_qa_eval.demo_llm import llm_demo_block  # noqa: E402

PASSAGES = [
    {"rank": 1, "title": "Alpha", "text": "TopicOne was a long project."},
    {"rank": 2, "title": "Beta", "text": "TopicOne was started by PersonOne in 1990."},
]
DOCS = Path(__file__).resolve().parents[1] / "docs"


def test_block_holds_the_answer_citations_and_the_passages_it_was_given() -> None:
    block = llm_demo_block("Who founded TopicOne?", PASSAGES, client=FakeAnswerer(), model="fake-model")
    assert block == {
        "model": "fake-model",
        "answer": "PersonOne",
        "citations": [2],
        "abstained": False,
        "valid": True,
        "passages": PASSAGES,
    }


def test_block_records_an_abstention() -> None:
    block = llm_demo_block("When did TopicNine end?", PASSAGES, client=FakeAnswerer(), model="fake-model")
    assert block["abstained"] is True and block["answer"] == "" and block["citations"] == []


def test_block_marks_an_unusable_reply_as_invalid_not_as_an_answer() -> None:
    block = llm_demo_block("Who founded TopicOne?", PASSAGES, client=FakeAnswerer(mode="junk"), model="m")
    assert block["valid"] is False and block["answer"] == "" and block["abstained"] is False


def test_committed_demo_examples_have_consistent_llm_blocks() -> None:
    path = DOCS / "data" / "examples.json"
    examples = json.loads(path.read_text(encoding="utf8"))["examples"]
    blocks = [ex["llm"] for ex in examples if "llm" in ex]
    if not blocks:
        pytest.skip("no precomputed LLM answers in the demo data yet (run `make demo-data`)")
    assert len(blocks) == len(examples)
    for b in blocks:
        ranks = [p["rank"] for p in b["passages"]]
        assert ranks == list(range(1, len(ranks) + 1))
        assert all(1 <= c <= len(ranks) for c in b["citations"])
        assert b["abstained"] == (b["valid"] and b["answer"] == "")
