from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from rag_qa_eval.evalcommon import write_run
from rag_qa_eval.report import render_blocks, replace_block, write_report


def test_replace_block_swaps_only_the_marked_region():
    text = "intro\n<!-- BLOCK:a:START -->\nold\n<!-- BLOCK:a:END -->\noutro\n"
    out = replace_block(text, "a", "new line")
    assert out == "intro\n<!-- BLOCK:a:START -->\nnew line\n<!-- BLOCK:a:END -->\noutro\n"
    assert replace_block(out, "a", "new line") == out  # idempotent


def test_replace_block_requires_both_markers():
    with pytest.raises(ValueError):
        replace_block("no markers here", "a", "x")
    with pytest.raises(ValueError):
        replace_block("<!-- BLOCK:a:START -->\nonly start", "a", "x")


def test_blocks_contain_the_measured_numbers_and_no_nan(world: dict[str, Any]):
    blocks = render_blocks(world["retrieval"], world["reader"], world["latency"])
    assert set(blocks) == {"summary", "ablation", "topk", "reader", "errors", "latency"}
    head = world["reader"]["headline"]["test"]
    f1 = head["f1"]
    assert f"{f1[0]:.3f} [{f1[1]:.3f}, {f1[2]:.3f}]" in blocks["summary"]
    for text in blocks.values():
        assert not re.search(r"\bnan\b", text.lower())
    assert "| category | count |" in blocks["errors"]
    assert "p50" in blocks["latency"]


def test_write_report_updates_readme_and_results_file(world: dict[str, Any], tmp_path: Path):
    runs = tmp_path / "runs"
    for stage in ("retrieval", "reader", "latency"):
        write_run(runs, stage, world[stage], "# log\n")
    readme = tmp_path / "README.md"
    names = ("summary", "ablation", "topk", "reader", "errors", "latency")
    readme.write_text(
        "\n".join(f"<!-- BLOCK:{n}:START -->\n<!-- BLOCK:{n}:END -->" for n in names) + "\n",
        encoding="utf8",
    )
    results = tmp_path / "results.md"
    write_report(runs, readme, results)
    body = readme.read_text(encoding="utf8")
    assert "| Component |" in body and "BLOCK:summary:START" in body
    assert results.read_text(encoding="utf8").startswith("# Results")
