from __future__ import annotations

import pytest

from rag_qa_eval.fusion import reciprocal_rank_fusion


def test_rrf_scores_follow_the_formula():
    fused = dict(reciprocal_rank_fusion([[1, 2, 3], [3, 1]], k=60))
    assert fused[1] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[2] == pytest.approx(1 / 62)
    assert fused[3] == pytest.approx(1 / 63 + 1 / 61)


def test_rrf_rewards_agreement_between_rankers():
    fused = reciprocal_rank_fusion([[10, 20, 30], [20, 40, 10]])
    assert fused[0][0] == 20  # ranked 2nd and 1st beats 1st and 3rd
    assert {d for d, _ in fused} == {10, 20, 30, 40}


def test_rrf_single_ranker_preserves_order():
    assert [d for d, _ in reciprocal_rank_fusion([[5, 4, 3]])] == [5, 4, 3]


def test_rrf_tie_break_is_deterministic():
    a = reciprocal_rank_fusion([[1, 2], [2, 1]])
    b = reciprocal_rank_fusion([[1, 2], [2, 1]])
    assert a == b
    assert [d for d, _ in a] == [1, 2]  # equal scores: lower id first


def test_rrf_top_n_truncates_and_empty_input_is_empty():
    assert len(reciprocal_rank_fusion([[1, 2, 3], [4, 5, 6]], top_n=2)) == 2
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_rejects_non_positive_k():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[1]], k=0)
