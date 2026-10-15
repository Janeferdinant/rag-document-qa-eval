"""End-to-end test of the evaluation stages with tiny fakes: no model downloads, no network."""

from __future__ import annotations

import json
import re
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rag_qa_eval.bm25 import tokenize
from rag_qa_eval.eval_latency import latency_markdown, run_latency
from rag_qa_eval.eval_reader import reader_markdown, run_reader
from rag_qa_eval.eval_retrieval import retrieval_markdown, run_retrieval, score_rankings
from rag_qa_eval.evalcommon import (
    Components,
    Settings,
    latest_run,
    prepare,
    write_run,
)
from rag_qa_eval.reader import ReaderOutput

N_ART, N_PARA = 8, 4


def make_squad() -> dict[str, Any]:
    arts = []
    for a in range(N_ART):
        paras = []
        for p in range(N_PARA):
            topic = f"Topic{a}x{p}"
            ctx = (
                f"{topic} was founded in the year {1800 + a * 10 + p} by Person{a}x{p}. "
                f"It is located in City{a}x{p} near the river. Many visitors come every summer "
                f"to see {topic} and enjoy the long walks along the water."
            )
            who = ctx.index(f"Person{a}x{p}")
            qas = [
                {
                    "id": f"a{a}p{p}q0",
                    "question": f"Who founded {topic}?",
                    "is_impossible": False,
                    "answers": [{"text": f"Person{a}x{p}", "answer_start": who}],
                },
                {
                    "id": f"a{a}p{p}q1",
                    "question": f"Who painted {topic}?",
                    "is_impossible": True,
                    "answers": [],
                },
            ]
            paras.append({"context": ctx, "qas": qas})
        arts.append({"title": f"Article{a}", "paragraphs": paras})
    return {"version": "v2.0", "data": arts}


class HashEncoder:
    model_name = "fake-hash"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(256, dtype=np.float32)
        for tok in tokenize(text):
            v[zlib.crc32(tok.encode()) % 256] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])


class OverlapReranker:
    def score(self, query: str, texts: list[str]) -> list[float]:
        q = set(tokenize(query))
        return [float(len(q & set(tokenize(t)))) for t in texts]


class FakeReader:
    """'Who founded X?' is answered from 'by PersonN' when the topic is in the passage."""

    def read_many(self, pairs: list[tuple[str, str]]) -> list[ReaderOutput]:
        outs = []
        for q, passage in pairs:
            topic = re.search(r"(Topic\w+)", q)
            m = re.search(r"by (Person\w+)", passage)
            has_topic = bool(topic and topic.group(1) in passage)
            if m and "founded" in q and has_topic:
                outs.append(ReaderOutput(m.group(1), m.start(1), m.end(1), 9.0, 1.0))
            elif m:
                outs.append(ReaderOutput(m.group(1), m.start(1), m.end(1), 1.0, 4.0))
            else:
                outs.append(ReaderOutput("", 0, 0, float("-inf"), 4.0))
        return outs


def components() -> Components:
    return Components(
        encoder=lambda name: HashEncoder(),
        reranker=lambda: OverlapReranker(),
        reader=lambda name: FakeReader(),
    )


def settings() -> Settings:
    return Settings(
        n_calib_articles=3,
        n_calib_questions=40,
        n_test_questions=60,
        strategies=("sliding", "sentence"),
        sizes=(20, 40),
        embedders=("fake-hash",),
        readers=("fake-reader",),
        reader_ks=(1, 3),
        n_boot=200,
        latency_queries=6,
    )


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    tmp = tmp_path_factory.mktemp("eval")
    squad = tmp / "dev.json"
    squad.write_text(json.dumps(make_squad()), encoding="utf8")
    s, comps = settings(), components()
    prep = prepare(s, tmp, squad_path=squad)
    msgs: list[str] = []
    retrieval = run_retrieval(prep, s, comps, tmp / "cache", msgs.append)
    reader = run_reader(prep, s, comps, tmp / "cache", retrieval, msgs.append)
    latency = run_latency(prep, s, comps, tmp / "cache", retrieval, reader, msgs.append)
    return {"tmp": tmp, "prep": prep, "retrieval": retrieval, "reader": reader, "latency": latency}


def test_score_rankings_computes_all_metrics():
    out = score_rankings([[3, 1, 2]], [{1}], ks=(1, 3))
    assert out["recall@1"][0] == 0.0 and out["recall@3"][0] == 1.0
    assert out["mrr@10"][0] == 0.5
    assert 0 < out["ndcg@10"][0] < 1


def test_article_split_is_disjoint(world: dict[str, Any]):
    prep = world["prep"]
    assert not set(prep.calib_titles) & set(prep.test_titles)
    assert len(prep.calib_titles) == 3 and len(prep.test_titles) == 5


def test_retrieval_stage_covers_the_grid_and_selects_on_calibration(world: dict[str, Any]):
    r = world["retrieval"]
    methods = {e["method"] for e in r["entries"]}
    assert methods == {"bm25", "dense", "hybrid", "hybrid+rerank"}
    # grid: 2 strategies x 2 sizes x 3 methods x 2 splits, plus the reranker at the best chunking
    assert len(r["entries"]) == 2 * 2 * 3 * 2 + 2
    assert "best_before_reranker" in r["selected"]
    assert r["selected"]["criterion"].endswith("calibration split")
    for e in r["entries"]:
        mean, lo, hi = e["metrics"]["mrr@10"]
        assert lo <= mean <= hi
    assert {p["metric"] for p in r["paired_test"]} == {"mrr@10", "recall@5"}


def test_easy_synthetic_questions_are_retrieved_well(world: dict[str, Any]):
    best = [e for e in world["retrieval"]["entries"] if e["split"] == "test" and e["method"] == "bm25"]
    assert max(e["metrics"]["recall@5"][0] for e in best) > 0.9


def test_reader_stage_reports_grid_oracle_and_error_analysis(world: dict[str, Any]):
    r = world["reader"]
    grid = r["readers"]["fake-reader"]["grid"]
    assert len(grid) == 4 * 2  # four retrievers x k in (1, 3)
    head = r["headline"]
    assert head["method"] in {"bm25", "dense", "hybrid", "hybrid+rerank"} and head["k"] in (1, 3)
    ea = r["error_analysis"]
    assert sum(ea["counts"].values()) == ea["n_questions"] == r["meta"]["n_test_questions"]
    assert all(len(e["id"]) == 8 for e in ea["examples"])
    assert r["readers"]["fake-reader"]["oracle"]["test"]["em"][0] > 0.9


def test_abstention_threshold_makes_the_fake_reader_abstain_on_unanswerable(world: dict[str, Any]):
    t = world["reader"]["headline"]["test"]
    assert t["abstain_precision"][0] > 0.9 and t["abstain_recall"][0] > 0.9
    assert t["supported_by_cited_chunk"][0] == 1.0  # extractive answers come from the chunk


def test_per_question_rows_are_anonymised_and_contain_no_question_text(world: dict[str, Any]):
    row = world["reader"]["per_question"][0]
    assert len(row["qid"]) == 8 and "question" not in row and "prediction" not in row


def test_latency_stage_reports_percentiles(world: dict[str, Any]):
    lat = world["latency"]
    assert set(lat["methods"]) == {"bm25", "dense", "hybrid", "hybrid+rerank"}
    for m in lat["methods"].values():
        assert m["end_to_end_ms"]["p95"] >= m["end_to_end_ms"]["p50"] >= 0


def test_run_logs_roundtrip_and_markdown_renders(world: dict[str, Any], tmp_path: Path):
    for stage, md_fn in (
        ("retrieval", retrieval_markdown),
        ("reader", reader_markdown),
        ("latency", latency_markdown),
    ):
        result = world[stage]
        jp, mp = write_run(tmp_path, stage, result, md_fn(result))
        assert jp.exists() and mp.read_text(encoding="utf8").startswith("# ")
        assert latest_run(tmp_path, stage)["meta"]["stage"] == stage
