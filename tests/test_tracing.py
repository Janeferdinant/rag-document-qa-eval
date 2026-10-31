"""Optional OpenTelemetry tracing of the LLM answer step. Off unless RAG_TRACING=1."""

from __future__ import annotations

import pytest

pytest.importorskip("evalharness")
pytest.importorskip("opentelemetry.sdk.trace")

from evalharness.llm.base import Completion  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402

from rag_qa_eval.llm_answers import AnswerResult  # noqa: E402
from rag_qa_eval.tracing import record_answer, setup_tracing  # noqa: E402


def result() -> AnswerResult:
    c = Completion(
        text="x",
        input_tokens=120,
        output_tokens=15,
        latency_s=0.4,
        model="qwen3:8b",
        cost_usd=0.002,
        billing="local",
    )
    return AnswerResult("Romulus", [1], abstained=False, valid=True, completions=[c])


def test_tracing_disabled_by_default() -> None:
    assert setup_tracing({}) is None
    assert setup_tracing({"RAG_TRACING": "0"}) is None
    # nothing is recorded and nothing fails when tracing is off
    record_answer(None, "Who founded Rome?", result())


def test_tracing_enabled_creates_spans_with_token_attributes() -> None:
    exporter = InMemorySpanExporter()
    tracer = setup_tracing({"RAG_TRACING": "1"}, exporter=exporter)
    assert tracer is not None
    record_answer(tracer, "Who founded Rome?", result())
    (span,) = exporter.get_finished_spans()
    attrs = dict(span.attributes or {})
    assert attrs["openinference.span.kind"] == "LLM" and attrs["llm.model_name"] == "qwen3:8b"
    assert attrs["llm.token_count.prompt"] == 120 and attrs["llm.token_count.completion"] == 15
    assert attrs["llm.token_count.total"] == 135 and attrs["llm.cost.total"] == pytest.approx(0.002)
    assert attrs["input.value"] == "Who founded Rome?" and attrs["output.value"] == "Romulus"


def test_invalid_answers_are_marked_in_the_trace() -> None:
    exporter = InMemorySpanExporter()
    tracer = setup_tracing({"RAG_TRACING": "1"}, exporter=exporter)
    bad = AnswerResult(None, [], abstained=False, valid=False, invalid_reason="citation 9", completions=[])
    record_answer(tracer, "q?", bad)
    (span,) = exporter.get_finished_spans()
    assert (span.attributes or {})["rag.answer.valid"] is False and not span.status.is_ok


def test_span_duration_matches_the_models_latency() -> None:
    exporter = InMemorySpanExporter()
    tracer = setup_tracing({"RAG_TRACING": "1"}, exporter=exporter)
    record_answer(tracer, "Who founded Rome?", result())  # the completion took 0.4 s
    (span,) = exporter.get_finished_spans()
    assert span.start_time is not None and span.end_time is not None
    assert (span.end_time - span.start_time) / 1e9 == pytest.approx(0.4, abs=0.01)
