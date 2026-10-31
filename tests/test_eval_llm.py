"""The `llm` evaluation stage on the tiny fake corpus (see conftest.py and fakes.py): no models, no network."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("evalharness")

from evalharness.llm.base import UsageLimitError  # noqa: E402
from evalharness.llm.cache import CachingClient  # noqa: E402

from llm_fakes import FakeAnswerer, n_test, run_fake  # noqa: E402
from rag_qa_eval.analysis import anonymise_id  # noqa: E402
from rag_qa_eval.eval_llm import stratified_sample  # noqa: E402


@pytest.fixture(scope="module")
def result(world: dict[str, Any], tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return run_fake(world, tmp_path_factory.mktemp("llm"))


def test_stratified_sample_is_seeded_and_keeps_the_answerable_share(world: dict[str, Any]) -> None:
    qs = world["prep"].sample("test", 60, 13, False)
    a = stratified_sample(qs, 20, seed=0)
    assert [q.qid for q in a] == [q.qid for q in stratified_sample(qs, 20, seed=0)]
    assert [q.qid for q in a] != [q.qid for q in stratified_sample(qs, 20, seed=1)]
    share = sum(not q.is_impossible for q in qs) / len(qs)
    assert abs(sum(not q.is_impossible for q in a) - share * 20) <= 1
    assert len(stratified_sample(qs, 500, seed=0)) == len(qs)
    order = {q.qid: i for i, q in enumerate(qs)}
    assert [order[q.qid] for q in a] == sorted(order[q.qid] for q in a)  # original order kept


def test_run_dict_has_metrics_cis_and_paired_deltas(result: dict[str, Any]) -> None:
    assert result["meta"]["stage"] == "llm"
    arm = result["arms"]["fake-local"]["sample"]
    for key in ("em", "f1", "abstain_precision", "abstain_recall", "citation_precision", "citation_recall"):
        mean, lo, hi = arm[key]
        assert lo <= mean <= hi, key
    assert arm["f1"][0] > 0.9 and arm["citation_precision"][0] == 1.0  # the fake answerer is easy to satisfy
    assert arm["invalid_rate"] == 0.0 and arm["latency_p50_s"] == pytest.approx(0.25)
    assert arm["tokens_in_per_1000"] == 100_000 and arm["cost_usd_per_1000"] == pytest.approx(1.0)
    for name in ("fake-local", "fake-hosted"):
        delta = result["paired"][name]["f1"]
        assert delta[1] <= delta[0] <= delta[2]
    assert set(result["baseline"]) >= {"sample", "all"}


def test_arms_are_scored_on_identical_question_ids(result: dict[str, Any], world: dict[str, Any]) -> None:
    local = [r["qid"] for r in result["per_question"]["fake-local"]]
    hosted = [r["qid"] for r in result["per_question"]["fake-hosted"]]
    sample_ids = {r["qid"] for r in result["per_question"]["fake-hosted"]}
    assert len(hosted) == 20 and sample_ids <= set(local) and len(local) == n_test(world)
    base = {r["qid"] for r in result["per_question"]["extractive-baseline"]}
    assert base == set(local)
    assert (
        result["arms"]["fake-local"]["sample"]["n_questions"]
        == result["arms"]["fake-hosted"]["sample"]["n_questions"]
    )


def test_per_question_rows_are_anonymised_and_hold_no_text(
    result: dict[str, Any], world: dict[str, Any]
) -> None:
    rows = result["per_question"]["fake-local"]
    ids = {anonymise_id(q.qid) for q in world["prep"].sample("test", 60, 13, False)}  # all test questions
    assert {r["qid"] for r in rows} == ids
    assert all("question" not in r and "answer" not in r for r in rows)


def test_invalid_answers_are_counted_in_the_run_not_dropped(world: dict[str, Any], tmp_path: Path) -> None:
    out = run_fake(world, tmp_path, lambda arm: FakeAnswerer(arm.provider, mode="junk"))
    arm = out["arms"]["fake-local"]["all"]
    assert arm["n_questions"] == n_test(world) and arm["invalid_rate"] == 1.0 and arm["f1"][0] == 0.0
    assert arm["abstain_recall"][0] == 0.0  # an invalid reply is not an abstention


def test_llm_stage_resumes_after_usage_limit_without_repeating_calls(
    world: dict[str, Any], tmp_path: Path
) -> None:
    first = FakeAnswerer("fake", limit_at=3)
    with pytest.raises(UsageLimitError):
        run_fake(world, tmp_path, lambda arm: CachingClient(first, tmp_path / "answers", salt=arm.name))
    made = first.calls  # two answered, the third raised
    assert made == 3
    second = FakeAnswerer("fake")
    run_fake(world, tmp_path, lambda arm: CachingClient(second, tmp_path / "answers", salt=arm.name))
    # the local arm needs one request per test question (2 are cached), the sample-only arm 20 more
    assert second.calls == n_test(world) - (made - 1) + 20


def test_run_logs_arm_versions_and_settings(result: dict[str, Any]) -> None:
    meta = result["meta"]
    assert meta["k"] == 5 and meta["seed"] == 13 and meta["sample"]["n"] == 20 and meta["sample"]["seed"] == 0
    assert len(meta["prompt_sha256"]) == 64 and meta["retriever"]["method"] in {
        "hybrid+rerank",
        "hybrid",
        "dense",
        "bm25",
    }
    arms = {a["name"]: a for a in meta["arms"]}
    assert arms["fake-local"]["model"] == "fake-model" and arms["fake-local"]["model_digest"] == "digest-1"
    assert (
        arms["fake-hosted"]["provider"] == "claude_code" and arms["fake-hosted"]["billing"] == "subscription"
    )


def test_faithfulness_is_judged_on_answered_questions_with_a_caveat(result: dict[str, Any]) -> None:
    arm = result["arms"]["fake-local"]["sample"]
    assert arm["faithfulness_mean"][0] == 2.0 and arm["faithfulness_n"] > 0
    judge = result["meta"]["judge"]
    assert judge["model"] == "qwen3:8b" and judge["calibrated_on"] == "answer correctness"
    assert judge["kappa"] == pytest.approx(0.835)


def test_error_analysis_counts_categories_and_gives_examples(
    result: dict[str, Any], world: dict[str, Any]
) -> None:
    ea = result["error_analysis"]
    assert sum(ea["counts"].values()) == n_test(world) and "missed_abstention" in ea["counts"]
    assert all(len(e["id"]) == 8 for e in ea["examples"])


def test_parse_arms_maps_names_to_providers_and_samples() -> None:
    from rag_qa_eval.eval_llm import parse_arms

    arms = parse_arms(["qwen3:4b", "claude_code", "codex_cli"])
    assert [(a.name, a.provider, a.sample) for a in arms] == [
        ("qwen3:4b", "ollama", "all"),
        ("claude-code", "claude_code", "sample"),
        ("codex-cli", "codex_cli", "sample"),
    ]
    assert arms[2].model == "gpt-6.1-sol" and arms[1].model == "default"
    with pytest.raises(ValueError, match="gemini"):
        parse_arms(["gemini"])
