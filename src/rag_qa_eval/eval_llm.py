"""LLM answering evaluation: retrieve the top 5 passages, ask a model for a cited answer, score it.

Arms are an extractive baseline (taken from the reader run, on the same questions), local models on all test
questions, and hosted models (command line tools) on a fixed stratified sample. Every arm that ran on all questions is
also scored on the sample, so the comparison is paired. Needs the optional `llm` extra.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from evalharness.cases import Case
from evalharness.llm.base import LLMClient
from evalharness.metrics import Output, rubric_judge
from evalharness.stats import paired_bootstrap_delta

from .analysis import anonymise_id, pick_examples
from .chunking import Chunk
from .data import Question, gold_spans
from .eval_retrieval import make_retriever
from .evalcommon import (
    Components,
    Prepared,
    Settings,
    chunk_corpus,
    hardware_info,
    settings_dict,
    today,
)
from .llm_answers import SYSTEM_PROMPT, AnswerResult, generate_answer
from .llm_metrics import citation_precision, citation_recall
from .llm_report import BASELINE, CATEGORIES, DESCRIPTIONS, llm_markdown, llm_tables  # noqa: F401
from .metrics import bootstrap_ci, bootstrap_ratio_ci
from .qa_metrics import answer_in_context, score_prediction, supported_by_chunk
from .retrieval import Retriever, chunk_contains_span
from .tracing import record_answer

K = 5
SAMPLE_SEED = 0
FAITHFULNESS_RUBRIC = (
    "Judge whether the answer is supported by the cited passages, on a scale from 0 to 2.\n"
    "2 = fully supported: the cited passages state what the answer says.\n"
    "1 = partly supported: the passages support part of the answer, or only imply it.\n"
    "0 = not supported: the passages do not say it, or contradict it.\n"
    "Judge only against the passages shown, not against what you know."
)
# The faithfulness judge was calibrated for a different task (grading answers against gold answers), which the
# report states next to every faithfulness number. Source: llm-eval-harness v0.1.0, README Summary.
JUDGE_CALIBRATIONS = {
    "qwen3:8b": {
        "calibrated_on": "answer correctness",
        "task": "100 SQuAD 2.0 answers graded against gold answers, labelled by the author",
        "kappa": 0.835,
        "kappa_ci": [0.682, 0.945],
        "source": "llm-eval-harness v0.1.0 README",
    }
}


@dataclass(frozen=True)
class ArmSpec:
    name: str
    model: str
    provider: str  # "ollama", "claude_code" or "codex_cli"
    sample: Literal["all", "sample"]  # "all": every test question; "sample": only the stratified sample


def stratified_sample(questions: Sequence[Question], n: int, seed: int) -> list[Question]:
    """A seeded sample of `n` questions that keeps the share of unanswerable ones, in the original order."""
    if n >= len(questions):
        return list(questions)
    rng = random.Random(seed)
    groups = {
        flag: sorted((q for q in questions if q.is_impossible == flag), key=lambda q: q.qid)
        for flag in (False, True)
    }
    total = len(questions)
    n_no = round(n * len(groups[True]) / total)
    chosen = {
        q.qid
        for flag, k in ((False, n - n_no), (True, n_no))
        for q in rng.sample(groups[flag], min(k, len(groups[flag])))
    }
    return [q for q in questions if q.qid in chosen]


def prompt_sha256() -> str:
    return hashlib.sha256((SYSTEM_PROMPT + f"|k={K}").encode()).hexdigest()


def categorize_llm(
    *, answerable: bool, valid: bool, abstained: bool, in_context: bool, f1: float, cited_ok: bool
) -> str:
    if not valid:
        return "invalid_output"
    if not answerable:
        return "correct_abstention" if abstained else "missed_abstention"
    if not in_context:
        return "retrieval_failure"
    if abstained:
        return "over_abstention"
    if f1 < 0.5:
        return "wrong_answer"
    return "correct" if cited_ok else "wrong_citation"


def _cost(res: AnswerResult) -> float | None:
    costs = [c.cost_usd for c in res.completions]
    return None if any(c is None for c in costs) else float(sum(c for c in costs if c is not None))


def _row(
    q: Question, res: AnswerResult, passages: Sequence[str], supports: Sequence[bool], golds: Sequence[str]
) -> dict[str, Any]:
    prediction = None if res.abstained or not res.valid else res.answer
    em, f1 = score_prediction(prediction, golds) if res.valid else (0.0, 0.0)
    in_ctx = answer_in_context(golds, passages) if golds else False
    cited = res.citations if res.valid and not res.abstained else []
    supported: float | None = None
    if prediction is not None and cited:
        supported = float(any(supported_by_chunk(prediction, passages[r - 1]) for r in cited))
    cited_ok = bool(cited) and any(supports[r - 1] for r in cited)
    return {
        "qid": anonymise_id(q.qid),
        "answerable": not q.is_impossible,
        "valid": res.valid,
        "abstained": res.abstained and res.valid,
        "em": em,
        "f1": round(f1, 4),
        "in_context": in_ctx,
        "cited": cited,
        "citation_precision": citation_precision(cited, supports),
        "citation_recall": citation_recall(cited, supports) if cited else None,
        "supported": supported,
        "latency_s": round(sum(c.latency_s for c in res.completions), 4),
        "tokens_in": sum(c.input_tokens for c in res.completions),
        "tokens_out": sum(c.output_tokens for c in res.completions),
        "cost_usd": _cost(res),
        "faithfulness": None,
        "category": categorize_llm(
            answerable=not q.is_impossible,
            valid=res.valid,
            abstained=res.abstained and res.valid,
            in_context=in_ctx,
            f1=f1,
            cited_ok=cited_ok,
        ),
        "_prediction": prediction,
        "_cited_text": " ".join(passages[cited[0] - 1].split())[:240] if cited else "",
    }


def _defined(rows: Sequence[dict[str, Any]], key: str) -> list[float]:
    return [float(r[key]) for r in rows if r.get(key) is not None]


def _ci(values: Sequence[float], n_boot: int, seed: int) -> list[float] | None:
    return [round(x, 4) for x in bootstrap_ci(values, n_boot, seed)] if len(values) else None


def summarise_rows(rows: Sequence[dict[str, Any]], n_boot: int, seed: int) -> dict[str, Any]:
    """Metrics with 95% bootstrap intervals over questions for one arm on one set of questions."""
    ans = np.asarray([r["answerable"] for r in rows], dtype=bool)
    no = ~ans
    # An unanswerable question has no gold span to cite, so citation precision and recall are over answerable ones.
    answerable_rows = [r for r in rows if r["answerable"]]
    abst = np.asarray([r["abstained"] for r in rows], dtype=float)
    em = np.asarray([r["em"] for r in rows], dtype=float)
    f1 = np.asarray([r["f1"] for r in rows], dtype=float)
    lat = np.asarray([r["latency_s"] for r in rows], dtype=float)
    costs = [r["cost_usd"] for r in rows]
    n = len(rows)
    precision = bootstrap_ratio_ci(abst * no, abst, n_boot, seed)
    recall = bootstrap_ratio_ci(abst * no, no.astype(float), n_boot, seed)
    p, r = precision[0], recall[0]
    out: dict[str, Any] = {
        "n_questions": n,
        "n_answerable": int(ans.sum()),
        "n_unanswerable": int(no.sum()),
        "em": [round(x, 4) for x in bootstrap_ci(em, n_boot, seed)],
        "f1": [round(x, 4) for x in bootstrap_ci(f1, n_boot, seed)],
        "hasans_f1": _ci(f1[ans].tolist(), n_boot, seed),
        "abstain_precision": [round(x, 4) for x in precision],
        "abstain_recall": [round(x, 4) for x in recall],
        "abstain_f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
        "citation_precision": _ci(_defined(answerable_rows, "citation_precision"), n_boot, seed),
        "citation_recall": _ci(_defined(answerable_rows, "citation_recall"), n_boot, seed),
        "supported_by_cited_passage": _ci(_defined(rows, "supported"), n_boot, seed),
        "invalid_rate": round(sum(1 for x in rows if not x["valid"]) / n, 4),
        "latency_p50_s": round(float(np.percentile(lat, 50)), 3) if lat.any() else None,
        "latency_p95_s": round(float(np.percentile(lat, 95)), 3) if lat.any() else None,
        "tokens_in_per_1000": round(sum(r["tokens_in"] for r in rows) / n * 1000),
        "tokens_out_per_1000": round(sum(r["tokens_out"] for r in rows) / n * 1000),
        "cost_usd_per_1000": None if any(c is None for c in costs) else round(sum(costs) / n * 1000, 4),
    }
    faith = _defined(rows, "faithfulness")
    out["faithfulness_mean"] = _ci(faith, n_boot, seed)  # 0 to 2
    out["faithfulness_n"] = len(faith)
    out["fully_supported_share"] = _ci([1.0 if v == 2 else 0.0 for v in faith], n_boot, seed)
    return out


def _paired(base: Sequence[dict[str, Any]], rows: Sequence[dict[str, Any]], n_boot: int) -> dict[str, Any]:
    """Mean [95% CI] of arm minus baseline on the same questions (rows are matched by question id)."""
    order = {r["qid"]: i for i, r in enumerate(base)}
    aligned = sorted(rows, key=lambda r: order[r["qid"]])
    return {
        key: [
            round(x, 4)
            for x in paired_bootstrap_delta(
                [r[key] for r in base], [r[key] for r in aligned], n_resamples=n_boot, seed=0
            )
        ]
        for key in ("f1", "em")
    }


def _baseline_rows(
    reader: dict[str, Any], qs: Sequence[Question], supports_top1: dict[str, bool]
) -> list[dict[str, Any]]:
    """Rows of the extractive reader (taken from the reader run) in the shape `summarise_rows` expects."""
    by_id = {r["qid"]: r for r in reader["per_question"]}
    rows = []
    for q in qs:
        r = by_id.get(anonymise_id(q.qid))
        if r is None:
            raise ValueError(
                "the reader run does not hold this question: run the reader and llm stages with the same seed "
                "and number of test questions"
            )
        answered = not r["abstained"]
        rows.append(
            {
                "qid": r["qid"], "answerable": r["answerable"], "valid": True, "abstained": r["abstained"],
                "em": r["em"], "f1": r["f1"], "in_context": r["in_context"],
                "citation_precision": float(supports_top1[q.qid]) if answered else None,
                "citation_recall": None, "supported": None, "latency_s": 0.0, "tokens_in": 0, "tokens_out": 0,
                "cost_usd": 0.0, "faithfulness": None,
            }
        )  # fmt: skip
    return rows


def llm_retriever(
    prep: Prepared, settings: Settings, comps: Components, cache_dir: Path, retrieval_result: dict[str, Any]
) -> tuple[list[Chunk], Retriever, str, str, int]:
    """The chunks and retriever the `llm` stage uses (the retrieval run's selected configuration), plus the
    method, chunking strategy and chunk size. The browser demo's export calls this too, so its prompts match."""
    sel = retrieval_result["selected"]
    strategy, size, method = sel["strategy"], sel["size"], sel["method"]
    embedder = sel["embedder"] or settings.embedders[0]
    chunks = chunk_corpus(prep.docs, strategy, size)
    retriever = make_retriever(comps, chunks, embedder, method == "hybrid+rerank", cache_dir, settings.depth)
    return chunks, retriever, method, strategy, size


# The context size the harness always sends to Ollama (its Ollama client defaults to 8192).
OLLAMA_NUM_CTX = 8192


def describe_processor(ps: dict[str, Any], model: str) -> str | None:
    """Where Ollama runs a loaded model, from its `/api/ps` reply: '100% GPU', 'CPU' or a split. None if not loaded."""
    for m in ps.get("models", []):
        if m.get("name") == model or m.get("model") == model:
            size, vram = int(m.get("size", 0)), int(m.get("size_vram", 0))
            if size <= 0:
                return None
            if vram >= size:
                return "100% GPU"
            if vram <= 0:
                return "CPU"
            gpu = round(100 * vram / size)
            return f"{gpu}% GPU, {100 - gpu}% CPU"
    return None


def ollama_base_url(host: str | None) -> str:
    """The URL to reach Ollama from the `OLLAMA_HOST` variable, which may be a bind address (0.0.0.0) or lack a port."""
    if not host:
        return "http://127.0.0.1:11434"
    if host.startswith("http"):
        return host
    name, _, port = host.partition(":")
    return f"http://{'127.0.0.1' if name in ('0.0.0.0', '::') else name}:{port or 11434}"


def ollama_processor(arm: ArmSpec) -> str | None:
    """Load the arm's model on the local Ollama server and ask where it runs. None for other providers or when the
    server cannot say. The model torch measures as 'CPU only' elsewhere in the run log is a different thing."""
    if arm.provider != "ollama":
        return None
    base = ollama_base_url(os.environ.get("OLLAMA_HOST"))
    try:
        warm = urllib.request.Request(
            f"{base}/api/generate",
            data=json.dumps(
                {"model": arm.model, "prompt": "", "keep_alive": "2m", "options": {"num_ctx": OLLAMA_NUM_CTX}}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(warm, timeout=120).read()  # noqa: S310 (a local server the user runs)
        with urllib.request.urlopen(f"{base}/api/ps", timeout=30) as resp:  # noqa: S310
            return describe_processor(json.loads(resp.read()), arm.model)
    except (OSError, ValueError):
        return None


def arm_client(arm: ArmSpec, cache_dir: Path) -> LLMClient:
    """The arm's model client behind the response cache that makes the stage resumable (needs the llm extra)."""
    from evalharness.llm.base import get_client
    from evalharness.llm.cache import CachingClient

    folder = cache_dir / "llm" / arm.name.replace(":", "-")
    return CachingClient(get_client(arm.provider), folder, salt=f"{arm.provider}:{arm.model}")


def run_llm(
    prep: Prepared,
    settings: Settings,
    comps: Components,
    cache_dir: Path,
    retrieval_result: dict[str, Any],
    reader_result: dict[str, Any],
    arms: Sequence[ArmSpec],
    client_factory: Callable[[ArmSpec], LLMClient],
    *,
    judge: tuple[LLMClient, str] | None = None,
    sample_n: int = 100,
    tracer: Any | None = None,
    processor_probe: Callable[[ArmSpec], str | None] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run every arm. A `UsageLimitError` from a client stops the run and is re-raised: finished answers are in the
    client's response cache (wrap clients in `CachingClient`), so a rerun makes only the missing calls."""
    t_start = time.perf_counter()
    chunks, retriever, method, strategy, size = llm_retriever(
        prep, settings, comps, cache_dir, retrieval_result
    )
    sel = retrieval_result["selected"]
    embedder = sel["embedder"] or settings.embedders[0]
    test = prep.sample("test", settings.n_test_questions, settings.seed, False)
    sample = stratified_sample(test, sample_n, SAMPLE_SEED)
    sample_ids = {q.qid for q in sample}
    ranked = retriever.search_batch([q.text for q in test], method, K)
    log(f"retrieved top {K} with {method} ({strategy}-{size}); {len(test)} questions, sample {len(sample)}")

    material: dict[str, tuple[list[str], list[bool], list[str]]] = {}
    top1: dict[str, bool] = {}
    for q, top in zip(test, ranked, strict=True):
        doc = prep.doc_by_title[q.title]
        spans = gold_spans(q, doc)
        passages = [chunks[p].text for p in top]
        supports = [chunk_contains_span(chunks[p], doc.doc_id, spans) for p in top]
        material[q.qid] = (passages, supports, [a.text for a in q.answers])
        top1[q.qid] = supports[0] if supports else False

    per_question: dict[str, list[dict[str, Any]]] = {}
    arm_meta: list[dict[str, Any]] = []
    for arm in arms:
        client = client_factory(arm)
        qs = test if arm.sample == "all" else sample
        rows: list[dict[str, Any]] = []
        digest = billing = None
        for i, q in enumerate(qs, start=1):
            passages, supports, golds = material[q.qid]
            res = generate_answer(q.text, passages, client=client, model=arm.model)
            rows.append(_row(q, res, passages, supports, golds))
            record_answer(tracer, q.text, res)
            for c in res.completions:
                digest, billing = c.model_digest or digest, c.billing
            if i % 100 == 0:
                log(f"  {arm.name}: {i}/{len(qs)}")
        if judge is not None:
            _judge_rows(rows, qs, material, judge)
        per_question[arm.name] = rows
        arm_meta.append(
            {"name": arm.name, "model": arm.model, "provider": arm.provider, "model_digest": digest,
             "billing": billing, "questions": len(qs),
             "processor": processor_probe(arm) if processor_probe else None}
        )  # fmt: skip
        log(f"{arm.name}: done ({len(rows)} questions)")
    per_question[BASELINE] = _baseline_rows(reader_result, test, top1)

    by_arm: dict[str, dict[str, Any]] = {}
    paired: dict[str, dict[str, Any]] = {}
    paired_all: dict[str, dict[str, Any]] = {}
    base_all = per_question[BASELINE]
    base_sample = [r for q, r in zip(test, base_all, strict=True) if q.qid in sample_ids]
    sample_anon = {anonymise_id(q) for q in sample_ids}
    by_arm[BASELINE] = {
        "all": summarise_rows(base_all, settings.n_boot, settings.seed),
        "sample": summarise_rows(base_sample, settings.n_boot, settings.seed),
    }
    for arm in arms:
        rows = per_question[arm.name]
        entry: dict[str, Any] = {}
        sample_rows = [r for r in rows if r["qid"] in sample_anon]
        if arm.sample == "all":
            entry["all"] = summarise_rows(rows, settings.n_boot, settings.seed)
        entry["sample"] = summarise_rows(sample_rows, settings.n_boot, settings.seed)
        by_arm[arm.name] = entry
        paired[arm.name] = _paired(base_sample, sample_rows, settings.n_boot)
        if arm.sample == "all":
            paired_all[arm.name] = _paired(base_all, rows, settings.n_boot)

    by_id = {anonymise_id(q.qid): q for q in test}
    error_arms: dict[str, Any] = {}
    for arm in arms:
        if arm.sample != "all":
            continue
        arm_rows = per_question[arm.name]
        examples = []
        for r in pick_examples(arm_rows, per_category=2, seed=settings.seed, categories=CATEGORIES):
            q = by_id[r["qid"]]
            examples.append(
                {"id": r["qid"], "category": r["category"], "question": q.text,
                 "gold": sorted({a.text for a in q.answers})[:2], "prediction": r["_prediction"],
                 "cited_excerpt": r["_cited_text"], "citations": r["cited"]}
            )  # fmt: skip
        error_arms[arm.name] = {
            "counts": {c: sum(1 for r in arm_rows if r["category"] == c) for c in CATEGORIES},
            "examples": examples,
        }
    for rows in per_question.values():
        for r in rows:
            r.pop("_prediction", None)
            r.pop("_cited_text", None)
    meta: dict[str, Any] = {
        "stage": "llm",
        "date": today(),
        "settings": settings_dict(settings),
        "hardware": hardware_info(),
        "wall_seconds": round(time.perf_counter() - t_start, 1),
        "seed": settings.seed,
        "k": K,
        "retriever": {"strategy": strategy, "size": size, "embedder": embedder, "method": method},
        "sample": {"n": len(sample), "seed": SAMPLE_SEED, "stratified_by": "answerable or unanswerable"},
        "n_test_questions": len(test),
        "prompt_sha256": prompt_sha256(),
        "arms": arm_meta,
        "baseline": reader_result["headline"]["reader"],
        "baseline_k": reader_result["headline"]["k"],
        "judge": None,
    }
    if judge is not None:
        meta["judge"] = {"model": judge[1], **JUDGE_CALIBRATIONS.get(judge[1], {"calibrated_on": "nothing"})}
    return {
        "meta": meta,
        "baseline": by_arm[BASELINE],
        "arms": {a.name: by_arm[a.name] for a in arms},
        "paired": paired,
        "paired_all": paired_all,
        "error_analysis": {"descriptions": DESCRIPTIONS, "arms": error_arms},
        "per_question": per_question,
    }


def _judge_rows(
    rows: list[dict[str, Any]],
    qs: Sequence[Question],
    material: dict[str, tuple[list[str], list[bool], list[str]]],
    judge: tuple[LLMClient, str],
) -> None:
    """Score support by the cited passages for every answered question (0 to 2)."""
    metric = rubric_judge(judge[0], judge[1], FAITHFULNESS_RUBRIC, (0, 2))
    for q, row in zip(qs, rows, strict=True):
        if row["abstained"] or not row["valid"] or not row["cited"]:
            continue
        passages = material[q.qid][0]
        case = Case(
            id=row["qid"],
            input={"question": q.text, "cited_passages": [passages[r - 1] for r in row["cited"]]},
        )
        try:
            row["faithfulness"] = metric.score(case, Output(text=row["_prediction"])).value
        except ValueError:  # an unreadable judge reply: left unscored, counted by faithfulness_n
            row["faithfulness"] = None


HOSTED_ARMS = {
    "claude_code": ArmSpec("claude-code", "default", "claude_code", "sample"),
    "codex_cli": ArmSpec("codex-cli", "gpt-6.1-sol", "codex_cli", "sample"),
}


def parse_arms(names: Sequence[str]) -> list[ArmSpec]:
    """Arm names from the command line: `name:tag` is a local Ollama model, `claude_code` and `codex_cli` are the
    hosted command line tools (sample only). There is no Gemini subscription arm (see llm-eval-harness NOTICE.md)."""
    arms = []
    for n in names:
        if n in HOSTED_ARMS:
            arms.append(HOSTED_ARMS[n])
        elif ":" in n:
            arms.append(ArmSpec(n, n, "ollama", "all"))
        else:
            raise ValueError(
                f"unknown arm {n!r}; use an Ollama model such as qwen3:8b, claude_code or codex_cli"
            )
    return arms
