"""Citation metrics for LLM answers. Abstention statistics reuse `qa_metrics.abstention_stats`."""

from __future__ import annotations

from collections.abc import Sequence


def _cited(cited_ranks: Sequence[int], n_passages: int) -> set[int]:
    cited = set(cited_ranks)
    for rank in sorted(cited):
        if not 1 <= rank <= n_passages:
            raise ValueError(f"citation rank {rank} is outside the {n_passages} retrieved passages")
    return cited


def citation_precision(cited_ranks: Sequence[int], supports: Sequence[bool]) -> float | None:
    """Share of the cited passages that contain a gold answer span. `supports[i]` is about passage i + 1.
    None when nothing was cited (precision is undefined)."""
    cited = _cited(cited_ranks, len(supports))
    if not cited:
        return None
    return sum(1 for rank in cited if supports[rank - 1]) / len(cited)


def citation_recall(cited_ranks: Sequence[int], supports: Sequence[bool]) -> float | None:
    """Share of the retrieved passages that contain a gold answer span which were cited.
    None when no retrieved passage supports the answer (recall is undefined)."""
    cited = _cited(cited_ranks, len(supports))
    supporting = {i + 1 for i, s in enumerate(supports) if s}
    if not supporting:
        return None
    return len(cited & supporting) / len(supporting)


def mean_defined(values: Sequence[float | None]) -> float | None:
    """Mean of the defined values; None when there are none."""
    defined = [v for v in values if v is not None]
    return sum(defined) / len(defined) if defined else None
