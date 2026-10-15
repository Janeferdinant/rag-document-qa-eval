"""Retrieval metrics (binary relevance) and bootstrap confidence intervals."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def recall_at_k(rels: Sequence[bool], k: int) -> float:
    """Hit-based recall@k: 1.0 if at least one relevant chunk is in the top k, else 0.0."""
    return 1.0 if any(rels[:k]) else 0.0


def reciprocal_rank(rels: Sequence[bool], k: int = 10) -> float:
    for rank, rel in enumerate(rels[:k], start=1):
        if rel:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(rels: Sequence[bool], k: int, n_relevant: int) -> float:
    """nDCG@k with binary gains. `n_relevant` is the number of relevant chunks in the corpus."""
    dcg = sum(1.0 / math.log2(i + 2) for i, rel in enumerate(rels[:k]) if rel)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(n_relevant, k)))
    return dcg / ideal if ideal > 0 else 0.0


def bootstrap_ci(
    values: np.ndarray | Sequence[float],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Percentile bootstrap over items. Returns (mean, lower, upper) for a (1 - alpha) interval."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    mean = float(v.mean())
    if np.all(v == v[0]):
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return mean, float(lo), float(hi)


def paired_bootstrap_diff(
    a: np.ndarray | Sequence[float],
    b: np.ndarray | Sequence[float],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Bootstrap interval for mean(a) - mean(b), resampling questions jointly (paired)."""
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("paired samples must have the same length")
    return bootstrap_ci(x - y, n_boot=n_boot, seed=seed, alpha=alpha)


def bootstrap_ratio_ci(
    numerator: np.ndarray | Sequence[float],
    denominator: np.ndarray | Sequence[float],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Bootstrap interval for sum(numerator) / sum(denominator) (e.g. abstention precision).

    Questions are resampled jointly; resamples with an empty denominator count as 0.
    """
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    if num.shape != den.shape or num.size == 0:
        raise ValueError("numerator and denominator must be non-empty and of equal length")
    if den.sum() == 0:
        return 0.0, 0.0, 0.0
    point = float(num.sum() / den.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, num.size, size=(n_boot, num.size))
    d = den[idx].sum(axis=1)
    ratios = np.where(d > 0, num[idx].sum(axis=1) / np.where(d > 0, d, 1.0), 0.0)
    lo, hi = np.quantile(ratios, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)
