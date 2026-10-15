"""Browser checks for docs/index.html at 390 px and 1280 px (Playwright, headless Chromium).

Run with `make e2e`. The Gemini API is never called: every request to generativelanguage.googleapis.com
is intercepted and answered by the test, and any other external request is aborted and fails the test.
Set UPDATE_SCREENSHOTS=1 to refresh docs/screenshots.
"""

from __future__ import annotations

import functools
import http.server
import json
import os
import threading
from pathlib import Path
from typing import Any

import pytest

sync_api = pytest.importorskip("playwright.sync_api")

DOCS = Path(__file__).resolve().parents[2] / "docs"
SHOTS = DOCS / "screenshots"
pytestmark = pytest.mark.e2e
WIDTHS = [390, 1280]
FAKE_KEY = "AIzaSyFAKE-KEY-for-tests-0123456789"
GEMINI = "generativelanguage.googleapis.com"


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture(scope="module")
def base_url():
    handler = functools.partial(_Quiet, directory=str(DOCS))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


@pytest.fixture(scope="module")
def examples() -> dict[str, Any]:
    return json.loads((DOCS / "data" / "examples.json").read_text(encoding="utf8"))


@pytest.fixture(scope="module")
def corpus() -> dict[str, Any]:
    return json.loads((DOCS / "data" / "sample_corpus.json").read_text(encoding="utf8"))


class Session:
    """A page plus everything it logged: console errors, page errors and requests."""

    def __init__(self, browser: Any, width: int, base_url: str) -> None:
        self.ctx = browser.new_context(viewport={"width": width, "height": 900})
        self.page = self.ctx.new_page()
        self.console: list[str] = []
        self.errors: list[str] = []
        self.gemini_requests: list[Any] = []
        self.blocked: list[str] = []
        self.reply: dict[str, Any] = {
            "status": 200,
            "body": {"candidates": [{"content": {"parts": [{"text": "x [1]"}]}}]},
        }
        self.base_url = base_url
        self.page.on("console", lambda m: self.console.append(f"{m.type}: {m.text}"))
        self.page.on("pageerror", lambda e: self.errors.append(str(e)))
        self.page.route("**/*", self._route)

    def _route(self, route: Any) -> None:
        url = route.request.url
        if url.startswith(self.base_url):
            route.continue_()
        elif GEMINI in url:
            self.gemini_requests.append(route.request)
            if self.reply.get("abort"):
                route.abort()
            else:
                route.fulfill(
                    status=self.reply["status"],
                    content_type="application/json",
                    headers={"access-control-allow-origin": "*", "access-control-allow-headers": "*"},
                    body=json.dumps(self.reply["body"]),
                )
        else:
            self.blocked.append(url)
            route.abort()

    def open(self) -> None:
        self.page.goto(self.base_url + "/index.html")
        self.page.wait_for_selector(".chip")

    def close(self) -> None:
        self.ctx.close()


@pytest.fixture()
def make_session(browser: Any, base_url: str):
    sessions: list[Session] = []

    def make(width: int) -> Session:
        s = Session(browser, width, base_url)
        sessions.append(s)
        s.open()
        return s

    yield make
    for s in sessions:
        s.close()


def assert_clean(s: Session) -> None:
    assert s.errors == []
    assert [m for m in s.console if m.startswith(("error", "warning"))] == []
    assert s.blocked == []


def no_horizontal_scroll(s: Session) -> bool:
    return bool(s.page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"))


def shot(s: Session, name: str, width: int) -> None:
    if os.environ.get("UPDATE_SCREENSHOTS"):
        SHOTS.mkdir(exist_ok=True)
        s.page.screenshot(path=str(SHOTS / f"{name}-{width}.png"), full_page=True)


@pytest.mark.parametrize("width", WIDTHS)
def test_loads_in_mock_mode_with_the_billing_notice(make_session, width: int):
    s = make_session(width)
    text = s.page.inner_text("body")
    assert "Mock mode" in s.page.inner_text("#mode-badge")
    assert "billed to your own Google account" in text and "paid service in some regions" in text
    assert "SQuAD 2.0" in s.page.inner_text("#attribution") and "CC BY-SA 4.0" in s.page.inner_text(
        "#attribution"
    )
    assert no_horizontal_scroll(s)
    assert s.gemini_requests == []
    assert_clean(s)


@pytest.mark.parametrize("width", WIDTHS)
def test_precomputed_examples_show_label_answer_and_citations(make_session, examples, width: int):
    s = make_session(width)
    chips = s.page.locator(".chip")
    assert chips.count() == len(examples["examples"]) >= 4
    for i, ex in enumerate(examples["examples"]):
        chips.nth(i).click()
        result = s.page.locator("#result")
        assert "Precomputed" in result.locator(".tag").inner_text()
        assert s.page.locator(".passage").count() == len(ex["retrieved"])
        if ex["answer"] is None:
            assert "abstained" in result.inner_text()
        else:
            assert ex["answer"] in result.locator(".answer").inner_text()
            assert result.locator(".passage.cited").count() == 1
        assert ex["outcome_label"] in result.inner_text()
        assert no_horizontal_scroll(s)
    shot(s, "mock-example", width)
    assert s.gemini_requests == []
    assert_clean(s)


@pytest.mark.parametrize("width", WIDTHS)
def test_without_a_key_a_typed_question_shows_retrieval_only(make_session, corpus, width: int):
    s = make_session(width)
    question = " ".join(corpus["chunks"][0]["text"].split()[:8])
    s.page.fill("#question", question)
    s.page.click("#ask")
    result = s.page.locator("#result")
    assert "Retrieval only" in result.locator(".tag").inner_text()
    assert 1 <= s.page.locator(".passage").count() <= 3
    assert "No key is set" in result.inner_text()
    assert s.gemini_requests == []
    assert_clean(s)


@pytest.mark.parametrize("width", WIDTHS)
def test_live_mode_sends_the_key_only_to_google_and_renders_a_cited_answer(make_session, corpus, width: int):
    s = make_session(width)
    s.reply["body"] = {"candidates": [{"content": {"parts": [{"text": "A mocked answer [2]"}]}}]}
    s.page.fill("#key-input", FAKE_KEY)
    s.page.click("#use-key")
    assert "Live mode" in s.page.inner_text("#mode-badge")
    assert s.page.input_value("#key-input") == ""  # the field is cleared once the key is in memory
    question = " ".join(corpus["chunks"][0]["text"].split()[:8])
    s.page.fill("#question", question)
    s.page.click("#ask")
    s.page.wait_for_selector("#result .answer")
    assert len(s.gemini_requests) == 1
    req = s.gemini_requests[0]
    assert req.method == "POST"
    assert req.url.startswith(f"https://{GEMINI}/v1beta/models/") and req.url.endswith(":generateContent")
    assert FAKE_KEY not in req.url
    assert req.headers["x-goog-api-key"] == FAKE_KEY
    body = json.loads(req.post_data)
    prompt = body["contents"][0]["parts"][0]["text"]
    assert f"Question: {question}" in prompt and "[1] " in prompt
    result = s.page.locator("#result")
    assert "A mocked answer" in result.locator(".answer").inner_text()
    assert result.locator(".tag").inner_text().startswith("Answer from gemini")
    if s.page.locator(".passage").count() >= 2:
        assert s.page.locator("#passage-2.cited").count() == 1
    # the key is nowhere: not in the page, storage, cookies, console or any non-Google request
    assert FAKE_KEY not in s.page.content()
    assert s.page.evaluate("JSON.stringify([localStorage, sessionStorage, document.cookie])") == '[{},{},""]'
    assert all(FAKE_KEY not in m for m in s.console)
    shot(s, "live-mocked", width)
    assert no_horizontal_scroll(s)
    # clearing the key returns to mock mode and stops all calls
    s.page.click("#clear-key")
    assert "Mock mode" in s.page.inner_text("#mode-badge")
    s.page.click("#ask")
    assert len(s.gemini_requests) == 1
    assert_clean(s)


@pytest.mark.parametrize("width", WIDTHS)
def test_unanswerable_reply_is_shown_as_an_abstention(make_session, corpus, width: int):
    s = make_session(width)
    s.reply["body"] = {"candidates": [{"content": {"parts": [{"text": "UNANSWERABLE"}]}}]}
    s.page.fill("#key-input", FAKE_KEY)
    s.page.click("#use-key")
    s.page.fill("#question", " ".join(corpus["chunks"][0]["text"].split()[:6]))
    s.page.click("#ask")
    s.page.wait_for_selector("#result .answer")
    assert "abstained" in s.page.locator("#result .answer").inner_text()
    assert s.page.locator(".passage.cited").count() == 0
    assert_clean(s)


@pytest.mark.parametrize("width", WIDTHS)
def test_api_errors_are_explained_without_leaking_the_key(make_session, corpus, width: int):
    s = make_session(width)
    s.reply = {"status": 403, "body": {"error": {"message": f"API key not valid: {FAKE_KEY}"}}}
    s.page.fill("#key-input", FAKE_KEY)
    s.page.click("#use-key")
    s.page.fill("#question", " ".join(corpus["chunks"][0]["text"].split()[:6]))
    s.page.click("#ask")
    s.page.wait_for_selector("#result .error")
    msg = s.page.locator("#result .error").inner_text()
    assert "rejected" in msg and FAKE_KEY not in s.page.content()
    # a failed network call is reported too
    s.reply = {"abort": True}
    s.page.click("#ask")
    s.page.wait_for_function(
        "document.querySelector('#result .error')?.textContent.includes('Could not reach')"
    )
    assert FAKE_KEY not in s.page.content()
    # the browser logs a failed request itself; that is the only console noise allowed here
    assert s.errors == [] and s.blocked == []
    assert all(FAKE_KEY not in m for m in s.console)
