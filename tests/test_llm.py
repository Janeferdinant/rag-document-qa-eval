from __future__ import annotations

import json

import pytest

from rag_qa_eval.llm import (
    GeminiClient,
    LLMError,
    LLMReader,
    MissingKeyError,
    MockLLMClient,
    build_prompt,
    parse_llm_answer,
)

KEY = "AIzaSy-test-key-123456"


class FakeTransport:
    """Records the request and returns a canned HTTP response. Never touches the network."""

    def __init__(self, status: int = 200, body: dict | None = None) -> None:
        self.status = status
        self.body = (
            body if body is not None else {"candidates": [{"content": {"parts": [{"text": "Normandy [1]"}]}}]}
        )
        self.requests: list[tuple[str, dict[str, str], bytes]] = []

    def __call__(self, url: str, headers: dict[str, str], body: bytes, timeout: float):
        self.requests.append((url, headers, body))
        return self.status, json.dumps(self.body).encode()


# ---- Gemini client (mocked transport) --------------------------------------------------------


def test_gemini_request_shape_and_key_placement():
    t = FakeTransport()
    client = GeminiClient(api_key=KEY, model="gemini-test", transport=t)
    assert client.generate("hello", temperature=0.0, max_output_tokens=64) == "Normandy [1]"
    ((url, headers, body),) = t.requests
    assert url == "https://generativelanguage.googleapis.com/v1beta/models/gemini-test:generateContent"
    assert KEY not in url  # the key travels in a header, never in the URL
    assert headers["x-goog-api-key"] == KEY
    payload = json.loads(body)
    assert payload["contents"][0]["parts"][0]["text"] == "hello"
    assert payload["generationConfig"] == {"temperature": 0.0, "maxOutputTokens": 64}


def test_gemini_reads_the_key_from_the_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    t = FakeTransport()
    GeminiClient(transport=t).generate("q")
    assert t.requests[0][1]["x-goog-api-key"] == KEY


def test_gemini_without_a_key_fails_before_any_request(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    t = FakeTransport()
    with pytest.raises(MissingKeyError):
        GeminiClient(transport=t)
    assert t.requests == []


def test_gemini_repr_and_errors_never_contain_the_key():
    client = GeminiClient(
        api_key=KEY,
        transport=FakeTransport(status=403, body={"error": {"message": f"API key not valid: {KEY}"}}),
    )
    assert KEY not in repr(client) and KEY not in str(client)
    with pytest.raises(LLMError) as exc:
        client.generate("q")
    assert KEY not in str(exc.value) and "403" in str(exc.value)


def test_gemini_joins_multiple_parts():
    t = FakeTransport(body={"candidates": [{"content": {"parts": [{"text": "a"}, {"text": "b"}]}}]})
    assert GeminiClient(api_key=KEY, transport=t).generate("q") == "ab"


def test_gemini_empty_or_blocked_response_is_an_error():
    for body in ({"candidates": []}, {"promptFeedback": {"blockReason": "SAFETY"}}, {}):
        with pytest.raises(LLMError):
            GeminiClient(api_key=KEY, transport=FakeTransport(body=body)).generate("q")


def test_gemini_rejects_non_json_responses():
    def bad(url, headers, body, timeout):
        return 200, b"<html>oops</html>"

    with pytest.raises(LLMError):
        GeminiClient(api_key=KEY, transport=bad).generate("q")


# ---- prompt and answer parsing ---------------------------------------------------------------


def test_build_prompt_numbers_passages_from_one():
    p = build_prompt("Where?", ["alpha text", "beta text"])
    assert "[1] alpha text" in p and "[2] beta text" in p and "Question: Where?" in p
    assert "UNANSWERABLE" in p


@pytest.mark.parametrize(
    ("raw", "text", "cites"),
    [
        ("Normandy [1]", "Normandy", [1]),
        ("Normandy [1, 3].", "Normandy", [1, 3]),
        ('  "Denmark" [2]\n', "Denmark", [2]),
        ("Answer: 911 [1][2]", "911", [1, 2]),
        ("UNANSWERABLE", None, []),
        ("unanswerable.", None, []),
        ("", None, []),
    ],
)
def test_parse_llm_answer(raw: str, text: str | None, cites: list[int]):
    parsed = parse_llm_answer(raw)
    assert parsed.text == text and parsed.citations == cites


# ---- mock client and reader ------------------------------------------------------------------


def test_mock_client_is_deterministic_and_records_calls():
    chunks = ["The Normans were in Normandy.", "Cats sleep a lot."]
    prompt = build_prompt("Where were the Normans?", chunks)
    m = MockLLMClient()
    a, b = m.generate(prompt), m.generate(prompt)
    assert a == b and "[1]" in a and len(m.calls) == 2


def test_mock_client_says_unanswerable_without_overlap():
    prompt = build_prompt("Who painted it?", ["Cats sleep a lot."])
    assert MockLLMClient().generate(prompt) == "UNANSWERABLE"


def test_mock_client_can_return_scripted_responses():
    m = MockLLMClient(responses=["Paris [2]", "UNANSWERABLE"])
    assert m.generate("x") == "Paris [2]"
    assert m.generate("x") == "UNANSWERABLE"
    assert m.generate("x") == "UNANSWERABLE"  # the last scripted response repeats


def test_llm_reader_returns_the_cited_chunk_and_text():
    reader = LLMReader(MockLLMClient(responses=["Normandy [2]"]))
    ans = reader.answer("Where?", ["x", "The Normans lived in Normandy."])
    assert ans.text == "Normandy" and ans.chunk_index == 1 and not ans.abstained


def test_llm_reader_abstains_on_unanswerable_and_on_out_of_range_citations():
    assert LLMReader(MockLLMClient(responses=["UNANSWERABLE"])).answer("q", ["c"]).abstained
    bad = LLMReader(MockLLMClient(responses=["Paris [7]"])).answer("q", ["c"])
    assert bad.text == "Paris" and bad.chunk_index is None  # answer kept, citation invalid


def test_llm_reader_without_chunks_abstains_without_calling_the_model():
    m = MockLLMClient()
    assert LLMReader(m).answer("q", []).abstained
    assert m.calls == []
