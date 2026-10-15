"""LLM answering: a small `LLMClient` interface, a Gemini client and a deterministic mock.

The Gemini client is only used when the user supplies their own API key (argument or the
`GEMINI_API_KEY` environment variable). The committed evaluation and the tests never call it:
tests inject a fake transport, and the evaluation uses the extractive reader.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from .bm25 import tokenize

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"  # listed at ai.google.dev/gemini-api/docs/models on 2026-10-15

Transport = Callable[[str, dict[str, str], bytes, float], tuple[int, bytes]]

PROMPT_HEADER = (
    "Answer the question using only the numbered passages below. "
    "Reply with the shortest span that fully answers it, copied from a passage, "
    "followed by the passage number in square brackets, for example: Normandy [2]. "
    "If the passages do not contain the answer, reply exactly: UNANSWERABLE"
)


class LLMError(RuntimeError):
    """The model call failed. Messages never contain the API key."""


class MissingKeyError(LLMError):
    """No API key was supplied."""


class LLMClient(Protocol):
    name: str

    def generate(self, prompt: str, *, temperature: float = 0.0, max_output_tokens: int = 256) -> str: ...


def _urllib_transport(url: str, headers: dict[str, str], body: bytes, timeout: float) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()
    except urllib.error.URLError as err:
        raise LLMError(f"network error: {err.reason}") from None


class GeminiClient:
    """Minimal Gemini `generateContent` client. The key is sent in a header, never in the URL."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_GEMINI_MODEL,
        transport: Transport | None = None,
        timeout: float = 60.0,
    ) -> None:
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise MissingKeyError("no Gemini API key: pass api_key or set GEMINI_API_KEY")
        self._key = key
        self.model = model
        self.name = f"gemini:{model}"
        self._transport: Transport = transport or _urllib_transport
        self._timeout = timeout

    def __repr__(self) -> str:
        return f"GeminiClient(model={self.model!r}, api_key='***')"

    def _scrub(self, text: str) -> str:
        return text.replace(self._key, "[redacted]")

    def generate(self, prompt: str, *, temperature: float = 0.0, max_output_tokens: int = 256) -> str:
        url = GEMINI_ENDPOINT.format(model=self.model)
        headers = {"Content-Type": "application/json", "x-goog-api-key": self._key}
        body = json.dumps(
            {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_output_tokens,
                },
            }
        ).encode()
        status, raw = self._transport(url, headers, body, self._timeout)
        try:
            data = json.loads(raw)
        except ValueError:
            raise LLMError(f"Gemini API returned a non-JSON response (HTTP {status})") from None
        if status != 200:
            message = data.get("error", {}).get("message", "request failed") if isinstance(data, dict) else ""
            raise LLMError(self._scrub(f"Gemini API error {status}: {message}"))
        candidates = data.get("candidates") or []
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        text = "".join(p.get("text", "") for p in parts)
        if not text:
            reason = data.get("promptFeedback", {}).get("blockReason", "empty response")
            raise LLMError(f"Gemini returned no text ({reason})")
        return text


@dataclass
class MockLLMClient:
    """Deterministic stand-in for tests: scripted replies, or a keyword-overlap heuristic."""

    responses: Sequence[str] | None = None
    name: str = "mock"
    calls: list[str] = field(default_factory=list)

    def generate(self, prompt: str, *, temperature: float = 0.0, max_output_tokens: int = 256) -> str:
        self.calls.append(prompt)
        if self.responses:
            return self.responses[min(len(self.calls), len(self.responses)) - 1]
        return self._heuristic(prompt)

    @staticmethod
    def _heuristic(prompt: str) -> str:
        m = re.search(r"^Question: (.*)$", prompt, flags=re.M)
        q = set(tokenize(m.group(1))) if m else set()
        best, best_n, best_overlap = "", 0, 0
        for pm in re.finditer(r"^\[(\d+)\] (.*)$", prompt, flags=re.M):
            for sent in re.split(r"(?<=[.!?])\s+", pm.group(2)):
                overlap = len(q & set(tokenize(sent)))
                if overlap > best_overlap:
                    best, best_n, best_overlap = sent.strip(), int(pm.group(1)), overlap
        return f"{best} [{best_n}]" if best_overlap else "UNANSWERABLE"


def build_prompt(question: str, chunks: Sequence[str]) -> str:
    passages = "\n".join(f"[{i}] {' '.join(c.split())}" for i, c in enumerate(chunks, start=1))
    return f"{PROMPT_HEADER}\n\nPassages:\n{passages}\n\nQuestion: {question}\nAnswer:"


@dataclass(frozen=True)
class ParsedAnswer:
    text: str | None
    citations: list[int]


_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def parse_llm_answer(raw: str) -> ParsedAnswer:
    """Extract the answer text and its [n] citations; UNANSWERABLE or empty output means abstain."""
    s = raw.strip()
    if not s or re.fullmatch(r"unanswerable\W*", s, flags=re.I):
        return ParsedAnswer(None, [])
    cites = [int(n) for grp in _CITATION.findall(s) for n in re.split(r"\s*,\s*", grp)]
    text = _CITATION.sub("", s)
    text = re.sub(r"^(answer\s*:\s*)", "", text.strip(), flags=re.I)
    text = text.strip().strip("\"'").strip().rstrip(".").strip()
    return ParsedAnswer(text or None, cites)


@dataclass(frozen=True)
class LLMAnswer:
    text: str | None
    chunk_index: int | None  # zero-based index of the cited chunk, when the citation is valid
    raw: str = ""

    @property
    def abstained(self) -> bool:
        return self.text is None


class LLMReader:
    """Answers from retrieved chunks with any `LLMClient`."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def answer(self, question: str, chunks: Sequence[str]) -> LLMAnswer:
        if not chunks:
            return LLMAnswer(None, None)
        raw = self.client.generate(build_prompt(question, chunks))
        parsed = parse_llm_answer(raw)
        if parsed.text is None:
            return LLMAnswer(None, None, raw)
        idx = next((c - 1 for c in parsed.citations if 1 <= c <= len(chunks)), None)
        return LLMAnswer(parsed.text, idx, raw)
