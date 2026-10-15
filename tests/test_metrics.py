from __future__ import annotations

import math

import numpy as np
import pytest

from rag_qa_eval.metrics import (
    bootstrap_ci,
    bootstrap_ratio_ci,
    ndcg_at_k,
    paired_bootstrap_diff,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k_is_a_hit_indicator():
    rels = [False, False, True, False]
    assert recall_at_k(rels, 2) == 0.0
    assert recall_at_k(rels, 3) == 1.0
    assert recall_at_k(rels, 10) == 1.0  # k larger than the list is fine


def test_reciprocal_rank_uses_the_first_relevant_within_k():
    assert reciprocal_rank([False, True, True], 10) == 0.5
    assert reciprocal_rank([False, False, True], 2) == 0.0  # outside the cutoff
    assert reciprocal_rank([], 10) == 0.0


def test_ndcg_is_one_for_an_ideal_ranking_and_zero_for_a_miss():
    assert ndcg_at_k([True, True, False], 10, n_relevant=2) == pytest.approx(1.0)
    assert ndcg_at_k([False, False], 10, n_relevant=2) == 0.0


def test_ndcg_hand_computed_value():
    # relevant at ranks 2 and 4, three relevant chunks exist in the corpus
    rels = [False, True, False, True]
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(rels, 10, n_relevant=3) == pytest.approx(dcg / idcg)


def test_ndcg_caps_the_ideal_list_at_k():
    assert ndcg_at_k([True], 1, n_relevant=50) == pytest.approx(1.0)


def test_ndcg_without_any_relevant_chunk_is_zero_not_nan():
    assert ndcg_at_k([False, False], 10, n_relevant=0) == 0.0


def test_bootstrap_ci_brackets_the_mean_and_is_seeded():
    values = np.array([0.0, 1.0] * 50)
    mean, lo, hi = bootstrap_ci(values, n_boot=2000, seed=3)
    assert mean == pytest.approx(0.5)
    assert lo < 0.5 < hi
    assert lo > 0.3 and hi < 0.7
    assert bootstrap_ci(values, n_boot=2000, seed=3) == (mean, lo, hi)


def test_bootstrap_ci_of_constant_values_is_degenerate():
    mean, lo, hi = bootstrap_ci(np.ones(20), n_boot=500, seed=0)
    assert mean == lo == hi == 1.0


def test_bootstrap_ci_narrows_with_more_data():
    rng = np.random.default_rng(0)
    small = bootstrap_ci(rng.integers(0, 2, 40).astype(float), seed=1)
    large = bootstrap_ci(rng.integers(0, 2, 4000).astype(float), seed=1)
    assert (large[2] - large[1]) < (small[2] - small[1])


def test_bootstrap_ci_rejects_empty_input():
    with pytest.raises(ValueError):
        bootstrap_ci(np.array([]))


def test_paired_bootstrap_detects_a_consistent_improvement():
    a = np.array([1.0] * 80 + [0.0] * 20)
    b = np.array([1.0] * 60 + [0.0] * 40)
    diff, lo, hi = paired_bootstrap_diff(a, b, n_boot=2000, seed=0)
    assert diff == pytest.approx(0.2)
    assert lo > 0


def test_paired_bootstrap_interval_covers_zero_for_identical_systems():
    a = np.random.default_rng(1).integers(0, 2, 200).astype(float)
    diff, lo, hi = paired_bootstrap_diff(a, a.copy(), n_boot=500, seed=0)
    assert diff == 0.0 and lo == 0.0 and hi == 0.0


def test_paired_bootstrap_requires_equal_length():
    with pytest.raises(ValueError):
        paired_bootstrap_diff(np.ones(3), np.ones(4))


def test_bootstrap_ratio_ci_matches_the_point_estimate_and_brackets_it():
    num = np.array([1.0] * 30 + [0.0] * 70)
    den = np.array([1.0] * 60 + [0.0] * 40)
    ratio, lo, hi = bootstrap_ratio_ci(num, den, n_boot=1000, seed=0)
    assert ratio == pytest.approx(0.5)
    assert lo < ratio < hi


def test_bootstrap_ratio_ci_with_an_empty_denominator_is_zero():
    assert bootstrap_ratio_ci(np.zeros(5), np.zeros(5), n_boot=50, seed=0) == (0.0, 0.0, 0.0)
