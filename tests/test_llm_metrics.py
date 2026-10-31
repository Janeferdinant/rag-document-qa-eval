"""Citation precision and recall, on hand-made cases with known values."""

from __future__ import annotations

import pytest

from rag_qa_eval.llm_metrics import citation_precision, citation_recall, mean_defined


def test_two_citations_one_supporting_gives_precision_half() -> None:
    assert citation_precision([1, 3], [True, False, False, False, False]) == 0.5


def test_citing_the_only_supporting_passage_gives_recall_one() -> None:
    assert citation_recall([2], [False, True, False, False, False]) == 1.0


def test_citing_none_of_two_supporting_passages_gives_recall_zero() -> None:
    assert citation_recall([3], [True, False, False, False, True]) == 0.0


def test_partial_recall() -> None:
    assert citation_recall([1], [True, False, True, False, False]) == 0.5


def test_no_citations_gives_undefined_precision() -> None:
    assert citation_precision([], [True, False]) is None


def test_no_supporting_passage_in_the_top_k_gives_undefined_recall() -> None:
    assert citation_recall([1, 2], [False, False, False]) is None


def test_precision_with_no_supporting_passage_is_zero() -> None:
    assert citation_precision([1], [False, False]) == 0.0


def test_duplicated_citations_count_once() -> None:
    assert citation_precision([1, 1, 2], [True, False]) == 0.5
    assert citation_recall([1, 1], [True, True]) == 0.5


def test_rank_outside_the_passages_is_an_error() -> None:
    with pytest.raises(ValueError, match="rank 4"):
        citation_precision([4], [True, False, True])


def test_mean_defined_ignores_undefined_values() -> None:
    assert mean_defined([1.0, None, 0.0]) == 0.5
    assert mean_defined([None, None]) is None
    assert mean_defined([]) is None
