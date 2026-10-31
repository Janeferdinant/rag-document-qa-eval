"""Rendering of the `llm` run log as Markdown tables (README block, run log, results page).

Pure formatting: it needs no optional dependency, so `rag-qa report` works on a base install.
"""

from __future__ import annotations

from typing import Any

from .evalcommon import fmt_ci

BASELINE = "extractive-baseline"
CATEGORIES = (
    "correct",
    "wrong_citation",
    "retrieval_failure",
    "over_abstention",
    "wrong_answer",
    "correct_abstention",
    "missed_abstention",
    "invalid_output",
)
DESCRIPTIONS = {
    "correct": "answerable, answer matches a gold answer and the cited passage contains it",
    "wrong_citation": "answerable, answer matches a gold answer but the cited passages do not contain a gold span",
    "retrieval_failure": "answerable, but no retrieved passage contains a gold answer string",
    "over_abstention": "answerable, a retrieved passage contains the answer, the model abstained",
    "wrong_answer": "answerable, the answer does not match a gold answer",
    "correct_abstention": "unanswerable, the model abstained",
    "missed_abstention": "unanswerable, the model gave an answer",
    "invalid_output": "no valid cited answer after one retry",
}
ACCESS = {
    "ollama": "local (Ollama)",
    "claude_code": "via Claude Code on a Team subscription",
    "codex_cli": "via Codex CLI on a ChatGPT Plus subscription",
}


def _c(cell: list[float] | None) -> str:
    return "n/a" if cell is None else fmt_ci(cell)


def _table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(out + ["| " + " | ".join(r) + " |" for r in rows])


def _arm_row(name: str, access: str, label: str, s: dict[str, Any], status: str) -> list[str]:
    cost = s["cost_usd_per_1000"]
    return [
        name,
        access,
        f"{label} (n={s['n_questions']})",
        _c(s["em"]),
        _c(s["f1"]),
        f"{s['abstain_precision'][0]:.3f} / {s['abstain_recall'][0]:.3f}",
        _c(s["citation_precision"]),
        _c(s["citation_recall"]),
        _c(s["faithfulness_mean"]),
        f"{s['invalid_rate']:.3f}",
        "-" if s["latency_p50_s"] is None else f"{s['latency_p50_s']:.2f} s",
        "-" if cost is None else f"${cost:.2f}",
        status,
    ]


def llm_tables(result: dict[str, Any]) -> str:
    """The arm table, the paired differences and the notes, as Markdown (run log and README block)."""
    meta = result["meta"]
    status = f"Verified (run {meta['date']}, seed {meta['seed']})"
    access = {a["name"]: ACCESS.get(a["provider"], a["provider"]) for a in meta["arms"]}
    rows = [
        _arm_row(BASELINE, "local, CPU", "sample", result["baseline"]["sample"], status),
        _arm_row(BASELINE, "local, CPU", "all test questions", result["baseline"]["all"], status),
    ]
    for name, entry in result["arms"].items():
        rows.append(_arm_row(name, access[name], "sample", entry["sample"], status))
        if "all" in entry:
            rows.append(_arm_row(name, access[name], "all test questions", entry["all"], status))
    header = [
        "Arm", "Access", "Questions", "Exact match", "F1", "Abstention precision / recall",
        "Citation precision", "Citation recall", "Faithfulness (0 to 2)", "Invalid", "p50 latency",
        "Cost per 1,000 (API-equivalent)", "Status",
    ]  # fmt: skip
    deltas = [[name, _c(d["f1"]), _c(d["em"])] for name, d in result["paired"].items()]
    deltas_all = [[name, _c(d["f1"]), _c(d["em"])] for name, d in result.get("paired_all", {}).items()]
    hosted = any(a["provider"] != "ollama" for a in meta["arms"])
    r = meta["retriever"]
    judge = meta.get("judge")
    if judge is None:
        caveat = "No faithfulness judge was run."
    else:
        calibration = ""
        if "kappa" in judge:
            calibration = (
                f" ({judge['task']}; weighted kappa {judge['kappa']}, 95% CI {judge['kappa_ci'][0]} to "
                f"{judge['kappa_ci'][1]}; {judge['source']})"
            )
        caveat = (
            f"Faithfulness is judged by {judge['model']} on answered questions (0 unsupported, 1 partly, "
            "2 fully supported by the cited passages). This judge was not calibrated for this rubric: its "
            f"calibration measured {judge.get('calibrated_on', 'something else')}{calibration}. "
            "Treat the faithfulness column as a rough check, not a validated measurement."
        )
    return (
        _table(header, rows)
        + "\n\n**Paired difference from the extractive baseline** on the same sample questions (mean [95% CI] of "
        "arm minus baseline, paired bootstrap):\n\n"
        + _table(["Arm", "F1 difference", "Exact match difference"], deltas)
        + "\n\n"
        + (
            f"The same differences on all {meta['n_test_questions']} test questions (arms that ran on all of them):"
            "\n\n" + _table(["Arm", "F1 difference", "Exact match difference"], deltas_all) + "\n\n"
            if deltas_all
            else ""
        )
        + f"Passages: the top {meta['k']} from {r['method']} over {r['strategy']} chunks of {r['size']} words "
        f"({r['embedder'].split('/')[-1]}). Prompt hash `{meta['prompt_sha256'][:12]}`. The sample is "
        f"{meta['sample']['n']} questions, stratified by answerable or unanswerable, seed {meta['sample']['seed']}. "
        + (
            "Local arms ran in Ollama on: "
            + ", ".join(f"{a['name']} {a['processor']}" for a in meta["arms"] if a.get("processor"))
            + ". "
            if any(a.get("processor") for a in meta["arms"])
            else ""
        )
        + f"The extractive baseline reads the top {meta.get('baseline_k', 1)} passage (the committed reader run), "
        "not five. Citation precision and recall are over answerable questions only (an unanswerable question has "
        "no gold span to cite); for the baseline, precision is whether its one passage contains a gold span. "
        "Latency is the time recorded for the call that produced each answer, so cached answers keep it. "
        + (
            "Hosted arms run through the vendors' command line tools, so they measure the model plus each tool's "
            "own system prompt, and their latency includes the tool's start-up. "
            if hosted
            else ""
        )
        + "Cost is the API-equivalent price of the tokens used, not money paid. "
        "Invalid answers (no valid cited answer after one retry) count as wrong. " + caveat
    )


def llm_markdown(result: dict[str, Any]) -> str:
    """The human-readable run log written next to the JSON."""
    meta, ea = result["meta"], result["error_analysis"]
    lines = [
        "# LLM answers with citations",
        "",
        f"- Date: {meta['date']}, seed {meta['seed']}, k {meta['k']}, prompt hash `{meta['prompt_sha256']}`",
        f"- Hardware: {meta['hardware']}",
        f"- Wall time: {meta['wall_seconds']} s; test questions {meta['n_test_questions']}, "
        f"sample {meta['sample']['n']}",
        "",
        "## Arms",
        "",
        llm_tables(result),
        "",
    ]
    for arm, entry in ea["arms"].items():
        lines += [
            f"## Error analysis ({arm})",
            "",
            _table(
                ["category", "count", "description"],
                [[c, str(entry["counts"][c]), ea["descriptions"][c]] for c in CATEGORIES],
            ),
            "",
        ]
        for e in entry["examples"]:
            lines.append(
                f"- `{e['id']}` {e['category']}: {e['question']} | gold {e['gold']} | model {e['prediction']!r} | "
                f"cited {e['citations']} | cited excerpt {e['cited_excerpt']!r}"
            )
        lines.append("")
    return "\n".join(lines) + "\n"
