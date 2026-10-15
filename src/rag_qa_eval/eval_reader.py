"""End-to-end answering evaluation: retrieval + extractive reader with abstention.

The abstention threshold and the (retriever, k) choice are tuned on the calibration questions
(articles disjoint from the test articles); numbers are reported on the test questions.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import (
    CATEGORIES,
    DESCRIPTIONS,
    anonymise_id,
    categorize,
    category_counts,
    pick_examples,
)
from .data import Question, gold_spans
from .eval_retrieval import make_retriever
from .evalcommon import (
    Components,
    Prepared,
    Settings,
    chunk_corpus,
    fmt_ci,
    hardware_info,
    settings_dict,
    today,
)
from .metrics import bootstrap_ci, bootstrap_ratio_ci
from .qa_metrics import answer_in_context, score_prediction, supported_by_chunk
from .reader import Answer, ReaderOutput, decide, tune_threshold
from .retrieval import METHODS, chunk_contains_span

ORACLE = "oracle-paragraph"


def read_all(
    reader: Any,
    pairs: dict[tuple[str, int], tuple[str, str]],
    cache: dict[tuple[str, int], ReaderOutput],
    log: Callable[[str], None],
) -> None:
    """Read every (question, passage) pair not yet in `cache`, longest passages together."""
    todo = [k for k in pairs if k not in cache]
    todo.sort(key=lambda k: len(pairs[k][1]))
    t0 = time.perf_counter()
    outs = reader.read_many([pairs[k] for k in todo])
    for k, o in zip(todo, outs, strict=True):
        cache[k] = o
    if todo:
        dt = time.perf_counter() - t0
        log(f"    read {len(todo)} passages in {dt:.1f}s ({dt / len(todo) * 1000:.1f} ms each)")


class Condition:
    """Reader outputs and context information for one split and one (retriever, k) choice."""

    def __init__(
        self,
        questions: Sequence[Question],
        outs: list[list[ReaderOutput]],
        ctx_texts: list[list[str]],
        ctx_has_span: list[list[bool]],
    ) -> None:
        self.questions, self.outs = questions, outs
        self.ctx_texts, self.ctx_has_span = ctx_texts, ctx_has_span

    def best(self) -> tuple[list[float], list[bool], list[Answer]]:
        """Best margin per question and the answer at threshold -inf (used for tuning)."""
        answers = [decide(o, float("-inf")) for o in self.outs]
        margins = [a.margin for a in answers]
        unans = [q.is_impossible for q in self.questions]
        return margins, unans, answers

    def f1_if_answered(self) -> list[float]:
        _, _, answers = self.best()
        return [
            0.0 if a.text is None else score_prediction(a.text, [x.text for x in q.answers])[1]
            for q, a in zip(self.questions, answers, strict=True)
        ]

    def predict(self, threshold: float) -> list[Answer]:
        return [decide(o, threshold) for o in self.outs]


def _finite(margins: list[float]) -> list[float]:
    return [m if np.isfinite(m) else -1e6 for m in margins]


def evaluate_condition(
    cond: Condition, threshold: float, n_boot: int, seed: int
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    qs = cond.questions
    answers = cond.predict(threshold)
    rows: list[dict[str, Any]] = []
    arr: dict[str, list[float]] = {
        k: []
        for k in (
            "em", "f1", "answerable", "abstained", "in_context", "supported", "cited_ok",
            "correct_answered",
        )
    }  # fmt: skip
    for i, (q, a) in enumerate(zip(qs, answers, strict=True)):
        golds = [x.text for x in q.answers]
        em, f1 = score_prediction(a.text, golds)
        in_ctx = answer_in_context(golds, cond.ctx_texts[i]) if golds else False
        supported = cited_ok = 0.0
        if a.text is not None and a.chunk_index is not None:
            supported = float(supported_by_chunk(a.text, cond.ctx_texts[i][a.chunk_index]))
            if f1 >= 0.5:
                cited_ok = float(cond.ctx_has_span[i][a.chunk_index])
        answered_good = float(a.text is not None and f1 >= 0.5)
        for k, v in zip(arr, (em, f1, float(not q.is_impossible), float(a.abstained), float(in_ctx),
                              supported, cited_ok, answered_good), strict=True):  # fmt: skip
            arr[k].append(v)
        rows.append(
            {
                "qid": q.qid,
                "answerable": not q.is_impossible,
                "abstained": a.abstained,
                "in_context": in_ctx,
                "em": em,
                "f1": round(f1, 4),
                "margin": None if not np.isfinite(a.margin) else round(a.margin, 3),
                "cited_rank": a.chunk_index,
                "prediction": a.text,
                "category": categorize(
                    answerable=not q.is_impossible, abstained=a.abstained, in_context=in_ctx,
                    em=em, f1=f1,
                ),
            }
        )  # fmt: skip
    A = {k: np.asarray(v) for k, v in arr.items()}
    has, no = A["answerable"] == 1, A["answerable"] == 0
    abst = A["abstained"]
    sup_den = (abst == 0).astype(float)
    cite_den = A["correct_answered"]
    precision = bootstrap_ratio_ci(abst * no, abst, n_boot, seed)
    recall = bootstrap_ratio_ci(abst * no, no.astype(float), n_boot, seed)
    p, r = precision[0], recall[0]
    f_abs = 2 * p * r / (p + r) if p + r else 0.0
    summary = {
        "n_questions": len(qs),
        "n_answerable": int(has.sum()),
        "n_unanswerable": int(no.sum()),
        "em": bootstrap_ci(A["em"], n_boot, seed),
        "f1": bootstrap_ci(A["f1"], n_boot, seed),
        "hasans_em": bootstrap_ci(A["em"][has], n_boot, seed) if has.any() else (0.0, 0.0, 0.0),
        "hasans_f1": bootstrap_ci(A["f1"][has], n_boot, seed) if has.any() else (0.0, 0.0, 0.0),
        "noans_accuracy": bootstrap_ci(abst[no], n_boot, seed) if no.any() else (0.0, 0.0, 0.0),
        "abstain_precision": precision,
        "abstain_recall": recall,
        "abstain_f1": (f_abs, f_abs, f_abs),
        "answer_in_context": bootstrap_ci(A["in_context"][has], n_boot, seed)
        if has.any()
        else (0.0, 0.0, 0.0),
        "supported_by_cited_chunk": bootstrap_ratio_ci(A["supported"], sup_den, n_boot, seed),
        "citation_correct_among_good_answers": bootstrap_ratio_ci(A["cited_ok"], cite_den, n_boot, seed),
    }  # fmt: skip
    summary = {k: ([round(x, 4) for x in v] if isinstance(v, tuple) else v) for k, v in summary.items()}
    return summary, rows, A


def build_conditions(
    prep: Prepared,
    qs: Sequence[Question],
    chunks: Sequence[Any],
    ranked: Sequence[Sequence[int]],
    k: int,
    cache: dict[tuple[str, int], ReaderOutput],
) -> Condition:
    outs, texts, has = [], [], []
    for q, r in zip(qs, ranked, strict=True):
        doc = prep.doc_by_title[q.title]
        spans = gold_spans(q, doc)
        top = list(r[:k])
        outs.append([cache[(q.qid, p)] for p in top])
        texts.append([chunks[p].text for p in top])
        has.append([chunk_contains_span(chunks[p], doc.doc_id, spans) for p in top])
    return Condition(qs, outs, texts, has)


def oracle_condition(
    prep: Prepared, qs: Sequence[Question], cache: dict[tuple[str, int], ReaderOutput]
) -> Condition:
    outs, texts, has = [], [], []
    for q in qs:
        para = _paragraph(prep, q)
        outs.append([cache[(q.qid, -1)]])
        texts.append([para])
        has.append([not q.is_impossible])
    return Condition(qs, outs, texts, has)


def _paragraph(prep: Prepared, q: Question) -> str:
    doc = prep.doc_by_title[q.title]
    return prep.data.articles[doc.doc_id].paragraphs[q.para_idx]


def run_reader(
    prep: Prepared,
    settings: Settings,
    comps: Components,
    cache_dir: Path,
    retrieval_result: dict[str, Any],
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    t_start = time.perf_counter()
    sel = retrieval_result["selected"]
    strategy, size = sel["strategy"], sel["size"]
    embedder = sel["embedder"] or settings.embedders[0]
    chunks = chunk_corpus(prep.docs, strategy, size)
    retriever = make_retriever(comps, chunks, embedder, True, cache_dir, settings.depth)
    kmax = max(settings.reader_ks)
    splits = {
        "calibration": prep.sample("calibration", settings.n_calib_questions, settings.seed, False),
        "test": prep.sample("test", settings.n_test_questions, settings.seed, False),
    }
    log(
        f"chunking {strategy}-{size}, embedder {embedder}; "
        f"questions: { ({s: len(q) for s, q in splits.items()}) }"
    )
    ranked = {
        s: {m: retriever.search_batch([q.text for q in qs], m, kmax) for m in METHODS}
        for s, qs in splits.items()
    }

    readers_out: dict[str, Any] = {}
    headline: dict[str, Any] = {}
    headline_rows: list[dict[str, Any]] = []
    oracle_fail: dict[str, bool] = {}
    for reader_idx, rname in enumerate(settings.readers):
        reader = comps.get_reader(rname)
        log(f"reader {rname}")
        first = reader_idx == 0
        methods = METHODS if first else (headline["method"],)
        combos = (
            [(m, k) for m in METHODS for k in settings.reader_ks]
            if first
            else [(headline["method"], headline["k"])]
        )
        cache: dict[tuple[str, int], ReaderOutput] = {}
        pairs: dict[tuple[str, int], tuple[str, str]] = {}
        for s, qs in splits.items():
            for q in qs:
                pairs[(q.qid, -1)] = (q.text, _paragraph(prep, q))
            for m in methods:
                for q, rk in zip(qs, ranked[s][m], strict=True):
                    for p in rk[:kmax]:
                        pairs[(q.qid, p)] = (q.text, chunks[p].text)
        read_all(reader, pairs, cache, log)

        grid: list[dict[str, Any]] = []
        calib_f1: dict[tuple[str, int], float] = {}
        for m, k in combos:
            cal = build_conditions(prep, splits["calibration"], chunks, ranked["calibration"][m], k, cache)
            margins, unans, _ = cal.best()
            thr = tune_threshold(_finite(margins), unans, cal.f1_if_answered())
            cal_sum, _, _ = evaluate_condition(cal, thr, settings.n_boot, settings.seed)
            tst = build_conditions(prep, splits["test"], chunks, ranked["test"][m], k, cache)
            tst_sum, rows, _ = evaluate_condition(tst, thr, settings.n_boot, settings.seed)
            calib_f1[(m, k)] = cal_sum["f1"][0]
            grid.append(
                {"method": m, "k": k, "threshold": round(thr, 3),
                 "calibration_f1": cal_sum["f1"][0], "test": tst_sum}
            )  # fmt: skip
            if first and (not headline or cal_sum["f1"][0] > headline["calibration_f1"]):
                headline = {
                    "reader": rname, "method": m, "k": k, "threshold": round(thr, 3),
                    "calibration_f1": cal_sum["f1"][0],
                    "criterion": "mean F1 on the calibration questions",
                }  # fmt: skip
                headline_rows = rows
            log(f"  {m:<14} k={k} thr={thr:+.2f} test EM={tst_sum['em'][0]:.3f} F1={tst_sum['f1'][0]:.3f}")

        ocal = oracle_condition(prep, splits["calibration"], cache)
        om, ou, _ = ocal.best()
        othr = tune_threshold(_finite(om), ou, ocal.f1_if_answered())
        osum, orows, _ = evaluate_condition(
            oracle_condition(prep, splits["test"], cache), othr, settings.n_boot, settings.seed
        )
        readers_out[rname] = {"grid": grid, "oracle": {"threshold": round(othr, 3), "test": osum}}
        log(f"  oracle paragraph: EM={osum['em'][0]:.3f} F1={osum['f1'][0]:.3f}")
        if first:
            oracle_fail = {r["qid"]: r["em"] < 1.0 for r in orows}

    head_grid = next(
        g
        for g in readers_out[headline["reader"]]["grid"]
        if (g["method"], g["k"]) == (headline["method"], headline["k"])
    )
    headline["test"] = head_grid["test"]
    headline["oracle_test"] = readers_out[headline["reader"]]["oracle"]["test"]

    counts = category_counts([r["category"] for r in headline_rows])
    reading = [
        r
        for r in headline_rows
        if r["category"].startswith("reading_failure") or r["category"] == "boundary_mismatch"
    ]
    also_fail = sum(1 for r in reading if oracle_fail[r["qid"]])
    by_qid = {q.qid: q for q in splits["test"]}
    pos = {q.qid: i for i, q in enumerate(splits["test"])}
    examples = []
    for r in pick_examples(headline_rows, per_category=3, seed=settings.seed):
        q = by_qid[r["qid"]]
        snippet = ""
        if r["cited_rank"] is not None:
            top = ranked["test"][headline["method"]][pos[q.qid]][: headline["k"]]
            snippet = " ".join(chunks[top[r["cited_rank"]]].text.split())[:240]
        examples.append(
            {
                "id": anonymise_id(q.qid),
                "category": r["category"],
                "question": q.text,
                "gold": [a.text for a in q.answers][:2],
                "prediction": r["prediction"],
                "cited_chunk_excerpt": snippet,
                "oracle_paragraph_correct": not oracle_fail[q.qid],
            }
        )
    analysis = {
        "counts": counts,
        "descriptions": DESCRIPTIONS,
        "n_questions": len(headline_rows),
        "n_answerable": sum(r["answerable"] for r in headline_rows),
        "answerable_retrieval_failures": counts["retrieval_failure"],
        "answerable_reading_failures": len(reading),
        "reading_failures_that_also_fail_with_oracle_paragraph": also_fail,
        "examples": examples,
    }
    keys = (
        "qid",
        "answerable",
        "abstained",
        "in_context",
        "em",
        "f1",
        "margin",
        "cited_rank",
        "category",
    )
    per_q = [{k: (anonymise_id(r["qid"]) if k == "qid" else r[k]) for k in keys} for r in headline_rows]
    return {
        "meta": {
            "stage": "reader",
            "date": today(),
            "settings": settings_dict(settings),
            "hardware": hardware_info(),
            "wall_seconds": round(time.perf_counter() - t_start, 1),
            "chunking": {"strategy": strategy, "size": size, "embedder": embedder},
            "n_calibration_questions": len(splits["calibration"]),
            "n_test_questions": len(splits["test"]),
        },
        "headline": headline,
        "readers": readers_out,
        "error_analysis": analysis,
        "per_question": per_q,
    }


def reader_markdown(result: dict[str, Any]) -> str:
    meta, head = result["meta"], result["headline"]
    ch = meta["chunking"]
    lines = [
        f"# Reader run, {meta['date']}",
        "",
        f"- Wall time: {meta['wall_seconds']} s on CPU. Seed: {meta['settings']['seed']}.",
        f"- Questions (answerable and unanswerable, natural mix): {meta['n_calibration_questions']} "
        f"calibration, {meta['n_test_questions']} test (disjoint articles).",
        f"- Chunking {ch['strategy']}-{ch['size']}; embedder {ch['embedder']}.",
        f"- Headline (chosen on calibration, {head['criterion']}): reader {head['reader']}, "
        f"retriever {head['method']}, top-{head['k']}, abstention threshold {head['threshold']}.",
        f"- Hardware: {meta['hardware']}",
        "",
    ]
    for rname, rd in result["readers"].items():
        lines += [f"## Reader {rname} (test split, mean [95% CI])", ""]
        lines += [
            "| retriever | k | thr | EM | F1 | HasAns F1 | NoAns acc | abst. prec | abst. rec | "
            "answer in ctx | cited chunk supports | cited chunk has gold |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for g in rd["grid"]:
            t = g["test"]
            lines.append(
                f"| {g['method']} | {g['k']} | {g['threshold']} | {fmt_ci(t['em'])} | {fmt_ci(t['f1'])} | "
                f"{fmt_ci(t['hasans_f1'])} | {fmt_ci(t['noans_accuracy'])} | "
                f"{fmt_ci(t['abstain_precision'])} | {fmt_ci(t['abstain_recall'])} | "
                f"{fmt_ci(t['answer_in_context'])} | {fmt_ci(t['supported_by_cited_chunk'])} | "
                f"{fmt_ci(t['citation_correct_among_good_answers'])} |"
            )
        o = rd["oracle"]["test"]
        lines += [
            "",
            f"Oracle paragraph (gold paragraph given to the reader, threshold {rd['oracle']['threshold']}): "
            f"EM {fmt_ci(o['em'])}, F1 {fmt_ci(o['f1'])}, abstention precision "
            f"{fmt_ci(o['abstain_precision'])}, recall {fmt_ci(o['abstain_recall'])}.",
            "",
        ]
    ea = result["error_analysis"]
    lines += [
        "## Error analysis (headline pipeline, test split)",
        "",
        "| category | count | meaning |",
        "|---|---|---|",
    ]
    for c in CATEGORIES:
        lines.append(f"| {c} | {ea['counts'][c]} | {ea['descriptions'][c]} |")
    lines += [
        "",
        f"Of {ea['answerable_reading_failures']} answerable questions with the answer in the retrieved "
        f"chunks that the reader got wrong, {ea['reading_failures_that_also_fail_with_oracle_paragraph']} "
        "also fail when the reader is given the gold paragraph.",
        "",
        "Examples (ids are hashed):",
        "",
    ]
    for e in ea["examples"]:
        lines.append(
            f"- `{e['id']}` {e['category']}: Q: {e['question']} | gold: {e['gold']} | "
            f"predicted: {e['prediction']!r} | oracle paragraph correct: {e['oracle_paragraph_correct']}"
        )
    return "\n".join(lines) + "\n"


def dump_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf8")
