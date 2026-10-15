"""End-to-end test of the evaluation stages with tiny fakes (see conftest.py and fakes.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rag_qa_eval.eval_latency import latency_markdown
from rag_qa_eval.eval_reader import reader_markdown
from rag_qa_eval.eval_retrieval import retrieval_markdown, score_rankings
from rag_qa_eval.evalcommon import latest_run, write_run


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
