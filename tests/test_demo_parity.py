"""The browser demo must behave like the Python code it mirrors (tokenizer, BM25, prompt, parsing)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from rag_qa_eval.bm25 import BM25Index, tokenize
from rag_qa_eval.llm import build_prompt, parse_llm_answer

DOCS = Path(__file__).resolve().parents[1] / "docs"
node = shutil.which("node")
pytestmark = pytest.mark.skipif(node is None, reason="node is not installed")

TEXTS = ["Café São Paulo 1999!", "The Rhine flows north.", "U.S. founders, e.g. Dr. Smith's team"]
CASES = [
    ("Normandy [2]", 2),
    ('Answer: "Denmark" [1, 3].', 3),
    ("UNANSWERABLE.", 2),
    ("  ", 2),
    ("The year 911 [9]", 2),
]


def run_node(payload: dict) -> dict:
    script = """
import { makeTokenizer, bm25Search, buildPrompt, parseAnswer, citedIndex } from './rag.js'
const p = JSON.parse(process.env.PAYLOAD)
const tok = makeTokenizer(p.stopwords)
const out = {
  tokens: p.texts.map((t) => tok(t)),
  prompt: buildPrompt(p.question, p.chunks),
  parsed: p.answers.map(([raw, n]) => { const r = parseAnswer(raw); return [r.text, r.citations, citedIndex(r.citations, n)] }),
  search: p.queries.map((q) => bm25Search(p.index, tok(q), 5)),
}
console.log(JSON.stringify(out))
"""
    import os

    env = {**os.environ, "PAYLOAD": json.dumps(payload)}
    res = subprocess.run(
        [node, "--input-type=module", "-e", script],
        cwd=DOCS,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(res.stdout)


def test_js_matches_python_for_tokenizer_prompt_parser_and_bm25():
    from rag_qa_eval.bm25 import STOPWORDS

    docs = [
        "The Rhine is a European river that begins in the Swiss canton of Graubunden.",
        "Prime numbers are natural numbers greater than 1 with no divisors other than 1 and themselves.",
        "The immune system protects an organism from disease. Cafe culture and the Rhine delta.",
    ]
    py_index = BM25Index([tokenize(d) for d in docs])
    queries = ["Where does the Rhine begin?", "What is a prime number?", "immune disease", "zzz"]
    payload = {
        "stopwords": sorted(STOPWORDS),
        "texts": TEXTS,
        "question": "Where?",
        "chunks": ["alpha  beta\n gamma", "delta"],
        "answers": [[c, n] for c, n in CASES],
        "queries": queries,
        "index": py_index.to_json(),
    }
    js = run_node(payload)
    assert js["tokens"] == [tokenize(t) for t in TEXTS]
    assert js["prompt"] == build_prompt("Where?", payload["chunks"])
    for (raw, n), (text, cites, idx) in zip(CASES, js["parsed"], strict=True):
        py = parse_llm_answer(raw)
        assert (text, cites) == (py.text, py.citations)
        expected_idx = next((c - 1 for c in py.citations if 1 <= c <= n), None)
        assert idx == expected_idx
    for q, hits in zip(queries, js["search"], strict=True):
        py_hits = py_index.search(tokenize(q), 5)
        assert [h[0] for h in hits] == [h[0] for h in py_hits]
        for (_, a), (_, b) in zip(hits, py_hits, strict=True):
            assert a == pytest.approx(b, rel=1e-9)


def test_demo_data_index_is_consistent_with_the_committed_corpus():
    corpus_path = DOCS / "data" / "sample_corpus.json"
    index_path = DOCS / "data" / "bm25_index.json"
    if not corpus_path.exists():
        pytest.skip("demo data not generated yet (run `make demo-data`)")
    corpus = json.loads(corpus_path.read_text(encoding="utf8"))
    index = json.loads(index_path.read_text(encoding="utf8"))
    assert index["index"]["n"] == len(corpus["chunks"])
    py = BM25Index([tokenize(c["text"]) for c in corpus["chunks"]])
    assert py.to_json()["doc_len"] == index["index"]["doc_len"]
    assert corpus["meta"]["attribution"]
