"""Reciprocal rank fusion (Cormack, Clarke and Buettcher, 2009)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]], k: int = 60, top_n: int | None = None
) -> list[tuple[int, float]]:
    """Fuse ranked id lists: score(d) = sum over rankers of 1 / (k + rank), rank starting at 1.

    Ties are broken by the lower id so results are deterministic.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    score: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            score[doc] += 1.0 / (k + rank)
    fused = sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))
    return fused[:top_n] if top_n is not None else fused
