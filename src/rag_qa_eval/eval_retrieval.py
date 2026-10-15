"""Retrieval evaluation: chunking x embedder x method grid with bootstrap confidence intervals.

Model selection uses the calibration split (articles disjoint from the test articles); the test
split is only used to report numbers.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .evalcommon import (
    Components,
    Prepared,
    Settings,
    chunk_corpus,
    doc_chunk_index,
    fmt_ci,
    hardware_info,
    question_relevant_chunks,
    settings_dict,
    today,
)
from .metrics import bootstrap_ci, ndcg_at_k, paired_bootstrap_diff, recall_at_k, reciprocal_rank
from .models import cached_embeddings
from .retrieval import METHODS, Retriever


def score_rankings(
    rankings: Sequence[Sequence[int]], rel_sets: Sequence[set[int]], ks: Sequence[int]
) -> dict[str, np.ndarray]:
    """Per-question metric arrays: recall@k (hit), MRR@10 and nDCG@10."""
    out: dict[str, list[float]] = {f"recall@{k}": [] for k in ks}
    out["mrr@10"], out["ndcg@10"] = [], []
    for ranked, rel in zip(rankings, rel_sets, strict=True):
        flags = [i in rel for i in ranked]
        for k in ks:
            out[f"recall@{k}"].append(recall_at_k(flags, k))
        out["mrr@10"].append(reciprocal_rank(flags, 10))
        out["ndcg@10"].append(ndcg_at_k(flags, 10, len(rel)))
    return {k: np.asarray(v) for k, v in out.items()}


def summarise(per_q: dict[str, np.ndarray], n_boot: int, seed: int) -> dict[str, list[float]]:
    return {m: [round(x, 4) for x in bootstrap_ci(v, n_boot, seed)] for m, v in per_q.items()}


def make_retriever(
    comps: Components,
    chunks: Sequence[Any],
    embedder: str | None,
    with_rerank: bool,
    cache_dir: Path,
    depth: int,
) -> Retriever:
    if embedder is None:
        return Retriever(chunks, candidate_depth=depth)
    enc = comps.get_encoder(embedder)
    emb = cached_embeddings(enc, [c.text for c in chunks], cache_dir)
    rr = comps.get_reranker() if with_rerank else None
    return Retriever(chunks, encoder=enc, reranker=rr, embeddings=emb, candidate_depth=100)


def run_retrieval(
    prep: Prepared,
    settings: Settings,
    comps: Components,
    cache_dir: Path,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    t_start = time.perf_counter()
    splits = {
        "calibration": prep.sample(
            "calibration", settings.n_calib_questions, settings.seed, answerable_only=True
        ),
        "test": prep.sample("test", settings.n_test_questions, settings.seed, answerable_only=True),
    }
    entries: list[dict[str, Any]] = []
    per_q_store: dict[tuple[str, int, str | None, str, str], dict[str, np.ndarray]] = {}
    chunk_counts: dict[str, int] = {}
    cut_rates: dict[str, dict[str, float]] = {}

    def evaluate_method(
        retriever: Retriever,
        rel: dict[str, list[set[int]]],
        strategy: str,
        size: int,
        emb: str | None,
        method: str,
        n_chunks: int,
    ) -> None:
        for split, qs in splits.items():
            t0 = time.perf_counter()
            ranked = retriever.search_batch([q.text for q in qs], method, settings.depth)
            per_q = score_rankings(ranked, rel[split], settings.ks)
            per_q_store[(strategy, size, emb, method, split)] = per_q
            entries.append(
                {
                    "strategy": strategy,
                    "size": size,
                    "embedder": emb,
                    "method": method,
                    "split": split,
                    "n_questions": len(qs),
                    "n_chunks": n_chunks,
                    "metrics": summarise(per_q, settings.n_boot, settings.seed),
                }
            )
            log(
                f"  {emb or '-':<42} {method:<14} {split:<11}"
                f" mrr@10={per_q['mrr@10'].mean():.3f}"
                f" r@5={per_q['recall@5'].mean():.3f} ({time.perf_counter() - t0:.1f}s)"
            )

    def relevance(chunks: list[Any]) -> dict[str, list[set[int]]]:
        by_doc = doc_chunk_index(chunks)
        return {
            s: [question_relevant_chunks(q, prep, chunks, by_doc) for q in qs] for s, qs in splits.items()
        }

    # Phase 1: chunking x embedder grid with BM25, dense and hybrid retrieval.
    for strategy in settings.strategies:
        for size in settings.sizes:
            chunks = chunk_corpus(prep.docs, strategy, size)
            key = f"{strategy}-{size}"
            chunk_counts[key] = len(chunks)
            rel = relevance(chunks)
            cut_rates[key] = {s: float(np.mean([not r for r in rs])) for s, rs in rel.items()}
            log(f"[{key}] {len(chunks)} chunks; span-cut rate test {cut_rates[key]['test']:.3f}")
            runs: list[tuple[str | None, str]] = [(None, "bm25")]
            for name in settings.embedders:
                runs += [(name, "dense"), (name, "hybrid")]
            retrievers: dict[str | None, Retriever] = {}
            for emb, method in runs:
                if emb not in retrievers:
                    t0 = time.perf_counter()
                    retrievers[emb] = make_retriever(comps, chunks, emb, False, cache_dir, settings.depth)
                    log(f"  built {emb or 'bm25'} index in {time.perf_counter() - t0:.1f}s")
                evaluate_method(retrievers[emb], rel, strategy, size, emb, method, len(chunks))

    def best_entry() -> dict[str, Any]:
        calib = [e for e in entries if e["split"] == "calibration"]
        return max(calib, key=lambda e: (e["metrics"]["ndcg@10"][0], e["metrics"]["mrr@10"][0]))

    # Phase 2: the cross-encoder reranker (slow on CPU) is evaluated at the best phase-1 chunking.
    best = best_entry()
    grid_best = {k: best[k] for k in ("strategy", "size", "embedder", "method")}
    rr_emb = best["embedder"] or settings.embedders[0]
    chunks = chunk_corpus(prep.docs, best["strategy"], best["size"])
    log(f"[rerank phase] {best['strategy']}-{best['size']} with {rr_emb}")
    rr_retriever = make_retriever(comps, chunks, rr_emb, True, cache_dir, settings.depth)
    evaluate_method(
        rr_retriever, relevance(chunks), best["strategy"], best["size"], rr_emb, "hybrid+rerank", len(chunks)
    )
    best = best_entry()
    selected = {k: best[k] for k in ("strategy", "size", "embedder", "method")}
    selected["criterion"] = "nDCG@10 on the calibration split"
    selected["best_before_reranker"] = grid_best

    # Paired comparisons on the test split at the selected chunking and embedder.
    s, z = grid_best["strategy"], grid_best["size"]
    emb_for_dense = rr_emb

    def pq(method: str) -> dict[str, np.ndarray] | None:
        emb = None if method == "bm25" else emb_for_dense
        return per_q_store.get((s, z, emb, method, "test"))

    pairs = [
        ("dense", "bm25"),
        ("hybrid", "bm25"),
        ("hybrid", "dense"),
        ("hybrid+rerank", "hybrid"),
        ("hybrid+rerank", "bm25"),
    ]
    paired: list[dict[str, Any]] = []
    for a, b in pairs:
        pa, pb = pq(a), pq(b)
        if pa is None or pb is None:
            continue
        for metric in ("mrr@10", "recall@5"):
            d, lo, hi = paired_bootstrap_diff(pa[metric], pb[metric], settings.n_boot, settings.seed)
            paired.append(
                {"a": a, "b": b, "metric": metric, "diff": round(d, 4), "lo": round(lo, 4),
                 "hi": round(hi, 4)}
            )  # fmt: skip

    return {
        "meta": {
            "stage": "retrieval",
            "date": today(),
            "settings": settings_dict(settings),
            "hardware": hardware_info(),
            "wall_seconds": round(time.perf_counter() - t_start, 1),
            "n_calibration_questions": len(splits["calibration"]),
            "n_test_questions": len(splits["test"]),
            "calibration_articles": prep.calib_titles,
            "n_test_articles": len(prep.test_titles),
            "n_documents": len(prep.docs),
            "corpus_words": sum(len(d.text.split()) for d in prep.docs),
        },
        "chunk_counts": chunk_counts,
        "span_cut_rate": cut_rates,
        "entries": entries,
        "selected": selected,
        "paired_test": paired,
    }


def retrieval_markdown(result: dict[str, Any]) -> str:
    meta, sel = result["meta"], result["selected"]
    lines = [
        f"# Retrieval run, {meta['date']}",
        "",
        f"- Wall time: {meta['wall_seconds']} s on CPU. Seed: {meta['settings']['seed']}.",
        f"- Corpus: {meta['n_documents']} Wikipedia articles ({meta['corpus_words']:,} words) from "
        "the SQuAD 2.0 dev set; calibration articles are disjoint from test articles.",
        f"- Questions (answerable only): {meta['n_calibration_questions']} calibration, "
        f"{meta['n_test_questions']} test. Bootstrap: {meta['settings']['n_boot']} resamples, 95%.",
        f"- Selected on calibration ({sel['criterion']}): {sel['method']}, {sel['strategy']} "
        f"chunks of {sel['size']} words, embedder {sel['embedder']}.",
        f"- Hardware: {meta['hardware']}",
        "",
        "Test split metrics (mean [95% CI]):",
        "",
        "| chunking | size | embedder | method | R@1 | R@5 | R@10 | MRR@10 | nDCG@10 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for e in result["entries"]:
        if e["split"] != "test":
            continue
        m = e["metrics"]
        lines.append(
            f"| {e['strategy']} | {e['size']} | {(e['embedder'] or '-').split('/')[-1]} | "
            f"{e['method']} | {fmt_ci(m['recall@1'])} | {fmt_ci(m['recall@5'])} | "
            f"{fmt_ci(m['recall@10'])} | {fmt_ci(m['mrr@10'])} | {fmt_ci(m['ndcg@10'])} |"
        )
    lines += ["", "Paired differences on the test split (selected chunking and embedder):", ""]
    lines += ["| a | b | metric | a - b [95% CI] |", "|---|---|---|---|"]
    for p in result["paired_test"]:
        lines.append(
            f"| {p['a']} | {p['b']} | {p['metric']} | {p['diff']:+.3f} [{p['lo']:+.3f}, {p['hi']:+.3f}] |"
        )
    lines += [
        "",
        "Share of answerable questions whose gold span is cut by every chunk boundary:",
        "",
    ]
    lines += ["| chunking-size | calibration | test |", "|---|---|---|"]
    for k, v in result["span_cut_rate"].items():
        lines.append(f"| {k} | {v['calibration']:.3f} | {v['test']:.3f} |")
    return "\n".join(lines) + "\n"


__all__ = ["METHODS", "run_retrieval", "retrieval_markdown", "score_rankings", "summarise"]
