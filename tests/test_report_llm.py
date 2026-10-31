"""The README block and the run log for the llm stage (needs the `llm` extra)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("evalharness")

from llm_fakes import run_fake  # noqa: E402
from rag_qa_eval.eval_llm import llm_markdown  # noqa: E402
from rag_qa_eval.evalcommon import write_run  # noqa: E402
from rag_qa_eval.report import render_blocks, write_report  # noqa: E402


@pytest.fixture(scope="module")
def llm_run(world: dict[str, Any], tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return run_fake(world, tmp_path_factory.mktemp("llm-report"))


def test_report_renders_llm_block_with_status_labels(world: dict[str, Any], llm_run: dict[str, Any]) -> None:
    blocks = render_blocks(world["retrieval"], world["reader"], world["latency"], llm_run)
    text = blocks["llm"]
    for name in ("extractive-baseline", "fake-local", "fake-hosted"):
        assert name in text
    assert f"Verified (run {llm_run['meta']['date']}, seed 13" in text
    f1 = llm_run["arms"]["fake-local"]["sample"]["f1"]
    assert f"{f1[0]:.3f} [{f1[1]:.3f}, {f1[2]:.3f}]" in text
    assert "via Claude Code on a Team subscription" in text and "local" in text
    assert "paired" in text.lower() and not re.search(r"\bnan\b", text.lower())
    assert set(blocks) >= {"summary", "llm"}


def test_report_states_the_judge_calibration_caveat(world: dict[str, Any], llm_run: dict[str, Any]) -> None:
    text = render_blocks(world["retrieval"], world["reader"], world["latency"], llm_run)["llm"]
    assert "answer correctness" in text and "0.835" in text and "not calibrated" in text.lower()


def test_report_without_an_llm_run_has_no_llm_block(world: dict[str, Any]) -> None:
    assert "llm" not in render_blocks(world["retrieval"], world["reader"], world["latency"])


def test_run_log_markdown_lists_arms_deltas_and_errors(llm_run: dict[str, Any]) -> None:
    md = llm_markdown(llm_run)
    assert md.startswith("# LLM answers") and "fake-local" in md and "missed_abstention" in md
    assert llm_run["meta"]["prompt_sha256"][:12] in md


def test_write_report_replaces_the_llm_block_and_extends_results(
    world: dict[str, Any], llm_run: dict[str, Any], tmp_path: Path
) -> None:
    runs = tmp_path / "runs"
    for stage in ("retrieval", "reader", "latency"):
        write_run(runs, stage, world[stage], "# log\n")
    write_run(runs, "llm", llm_run, llm_markdown(llm_run))
    names = ("summary", "ablation", "topk", "reader", "errors", "latency", "llm")
    readme = tmp_path / "README.md"
    readme.write_text(
        "\n".join(f"<!-- BLOCK:{n}:START -->\n<!-- BLOCK:{n}:END -->" for n in names) + "\n", "utf8"
    )
    results = tmp_path / "results.md"
    write_report(runs, readme, results)
    assert "fake-local" in readme.read_text(encoding="utf8")
    assert "LLM answers" in results.read_text(encoding="utf8")
