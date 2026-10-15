"""Per-query latency on CPU: retrieval stages and the extractive reader, one query at a time."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .eval_retrieval import make_retriever
from .evalcommon import Components, Prepared, Settings, chunk_corpus, hardware_info, today
from .retrieval import METHODS


def _pct(values: list[float], q: float) -> float:
    return round(float(np.percentile(values, q)), 2) if values else 0.0


def run_latency(
    prep: Prepared,
    settings: Settings,
    comps: Components,
    cache_dir: Path,
    retrieval_result: dict[str, Any],
    reader_result: dict[str, Any] | None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    sel = retrieval_result["selected"]
    embedder = sel["embedder"] or settings.embedders[0]
    chunks = chunk_corpus(prep.docs, sel["strategy"], sel["size"])
    retriever = make_retriever(comps, chunks, embedder, True, cache_dir, settings.depth)
    reader = comps.get_reader(settings.readers[0])
    k = reader_result["headline"]["k"] if reader_result else 3
    qs = prep.sample("test", settings.latency_queries, settings.seed + 1, answerable_only=False)
    warm = qs[:5]
    for m in METHODS:
        for q in warm:
            retriever.search(q.text, m, k)
    for q in warm:
        reader.read_many([(q.text, chunks[0].text)])
    rows: dict[str, Any] = {}
    for m in METHODS:
        stages: dict[str, list[float]] = {}
        totals_ret: list[float] = []
        totals_e2e: list[float] = []
        reads: list[float] = []
        for q in qs:
            t0 = time.perf_counter()
            ids, timings = retriever.search_timed(q.text, m, k)
            ret_ms = (time.perf_counter() - t0) * 1000
            for name, v in timings.items():
                stages.setdefault(name, []).append(v)
            t1 = time.perf_counter()
            reader.read_many([(q.text, chunks[i].text) for i in ids])
            read_ms = (time.perf_counter() - t1) * 1000
            totals_ret.append(ret_ms)
            reads.append(read_ms)
            totals_e2e.append(ret_ms + read_ms)
        rows[m] = {
            "retrieval_ms": {"p50": _pct(totals_ret, 50), "p95": _pct(totals_ret, 95)},
            "reader_ms": {"p50": _pct(reads, 50), "p95": _pct(reads, 95)},
            "end_to_end_ms": {"p50": _pct(totals_e2e, 50), "p95": _pct(totals_e2e, 95)},
            "stages_p50_ms": {name: _pct(v, 50) for name, v in stages.items()},
        }
        log(
            f"  {m:<14} retrieval p50 {rows[m]['retrieval_ms']['p50']} ms, "
            f"reader p50 {rows[m]['reader_ms']['p50']} ms"
        )
    return {
        "meta": {
            "stage": "latency",
            "date": today(),
            "hardware": hardware_info(),
            "n_queries": len(qs),
            "k": k,
            "n_chunks": len(chunks),
            "chunking": {"strategy": sel["strategy"], "size": sel["size"], "embedder": embedder},
            "note": "batch size 1, sequential queries, warm models, default torch threads",
        },
        "methods": rows,
    }


def latency_markdown(result: dict[str, Any]) -> str:
    meta = result["meta"]
    lines = [
        f"# Latency run, {meta['date']}",
        "",
        f"- {meta['n_queries']} test questions, top-{meta['k']} chunks read, {meta['n_chunks']} chunks "
        f"in the index ({meta['chunking']}). {meta['note']}.",
        f"- Hardware: {meta['hardware']}",
        "",
        "| retriever | retrieval p50 / p95 (ms) | reader p50 / p95 (ms) | end-to-end p50 / p95 (ms) |",
        "|---|---|---|---|",
    ]
    for m, r in result["methods"].items():
        a, b, c = r["retrieval_ms"], r["reader_ms"], r["end_to_end_ms"]
        lines.append(f"| {m} | {a['p50']} / {a['p95']} | {b['p50']} / {b['p95']} | {c['p50']} / {c['p95']} |")
    lines += ["", "Median per-stage retrieval time (ms):", ""]
    for m, r in result["methods"].items():
        lines.append(f"- {m}: {r['stages_p50_ms']}")
    return "\n".join(lines) + "\n"
