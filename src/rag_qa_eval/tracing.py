"""Optional OpenTelemetry tracing of the LLM answer step, with OpenInference attribute names.

Off unless the environment variable `RAG_TRACING=1`. Needs `pip install ".[llm,tracing]"`. Spans go to an OTLP
endpoint (default `http://localhost:6006/v1/traces`, where a local `phoenix serve` listens); no account or key.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from .llm_answers import AnswerResult

DEFAULT_ENDPOINT = "http://localhost:6006/v1/traces"


def setup_tracing(env: Mapping[str, str] | None = None, exporter: Any | None = None) -> Any | None:
    """A tracer when tracing is switched on, otherwise None. `exporter` replaces the OTLP exporter (tests)."""
    env = os.environ if env is None else env
    if env.get("RAG_TRACING") != "1":
        return None
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    if exporter is None:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter(endpoint=env.get("RAG_TRACING_ENDPOINT", DEFAULT_ENDPOINT))
    provider = TracerProvider(resource=Resource.create({"service.name": "rag-document-qa-eval"}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("rag_qa_eval")


def record_answer(tracer: Any | None, question: str, result: AnswerResult) -> None:
    """One span per answered question, with the model, token counts and API-equivalent cost. A no-op without a tracer."""
    if tracer is None:
        return
    from opentelemetry.trace import Status, StatusCode

    tokens_in = sum(c.input_tokens for c in result.completions)
    tokens_out = sum(c.output_tokens for c in result.completions)
    costs = [c.cost_usd for c in result.completions]
    with tracer.start_as_current_span("answer_question") as span:
        span.set_attribute("openinference.span.kind", "LLM")
        span.set_attribute("input.value", question)
        span.set_attribute("output.value", result.answer or "")
        if result.completions:
            span.set_attribute("llm.model_name", result.completions[-1].model)
        span.set_attribute("llm.token_count.prompt", tokens_in)
        span.set_attribute("llm.token_count.completion", tokens_out)
        span.set_attribute("llm.token_count.total", tokens_in + tokens_out)
        if costs and all(c is not None for c in costs):
            span.set_attribute("llm.cost.total", float(sum(c for c in costs if c is not None)))
        span.set_attribute("rag.answer.valid", result.valid)
        span.set_attribute("rag.answer.abstained", result.abstained)
        span.set_attribute("rag.answer.citations", [str(n) for n in result.citations])
        if not result.valid:
            span.set_status(Status(StatusCode.ERROR, result.invalid_reason or "invalid answer"))
