from __future__ import annotations

import pytest

from rag_qa_eval.qa_metrics import (
    abstention_stats,
    answer_in_context,
    exact_match,
    f1_score,
    normalize_answer,
    score_prediction,
    supported_by_chunk,
)


def test_normalize_answer_follows_the_squad_rules():
    assert normalize_answer("The  Eiffel Tower!") == "eiffel tower"
    assert normalize_answer("A, an, THE") == ""
    assert normalize_answer("U.S.A.") == "usa"


def test_exact_match_takes_the_best_gold_answer():
    assert exact_match("Denmark", ["Norway", "denmark"]) == 1.0
    assert exact_match("Denmark", ["Norway"]) == 0.0


def test_f1_partial_overlap_hand_computed():
    # prediction has 3 tokens, gold has 2, 2 shared: p = 2/3, r = 1 -> f1 = 0.8
    assert f1_score("the big red dog", ["red dog"]) == pytest.approx(0.8)
    assert f1_score("cat", ["dog"]) == 0.0


def test_f1_uses_token_multiplicity():
    assert f1_score("x b b", ["b b b"]) == pytest.approx(2 / 3)


def test_unanswerable_questions_are_scored_by_abstention():
    assert score_prediction(None, []) == (1.0, 1.0)  # correctly abstained
    assert score_prediction("Paris", []) == (0.0, 0.0)  # answered an unanswerable question
    assert score_prediction(None, ["Paris"]) == (0.0, 0.0)  # abstained on an answerable one


def test_answerable_questions_are_scored_against_the_gold_list():
    em, f1 = score_prediction("Paris", ["Paris", "the city of Paris"])
    assert em == 1.0 and f1 == 1.0


def test_abstention_precision_and_recall():
    # predicted abstain: T T F F ; truly unanswerable: T F T F
    p, r, f = abstention_stats([True, True, False, False], [True, False, True, False])
    assert p == pytest.approx(0.5) and r == pytest.approx(0.5) and f == pytest.approx(0.5)


def test_abstention_stats_degenerate_cases():
    assert abstention_stats([False, False], [True, False]) == (0.0, 0.0, 0.0)  # never abstains
    p, r, _ = abstention_stats([True, True], [False, False])
    assert p == 0.0 and r == 0.0  # no unanswerable questions: recall is defined as 0


def test_abstention_stats_requires_matching_lengths():
    with pytest.raises(ValueError):
        abstention_stats([True], [True, False])


def test_answer_in_context_normalises_before_matching():
    assert answer_in_context(["the Normans"], ["They called them Normans in 1066."])
    assert not answer_in_context(["Vikings"], ["They called them Normans."])
    assert not answer_in_context([], ["anything"])  # unanswerable: no gold answer to find


def test_answer_in_context_respects_token_boundaries():
    assert not answer_in_context(["cat"], ["concatenate the category"])


def test_supported_by_chunk_checks_the_cited_chunk_only():
    chunk = "The treaty was signed in 911 by Charles the Simple."
    assert supported_by_chunk("911", chunk)
    assert supported_by_chunk("signed in 911", chunk)
    assert not supported_by_chunk("1066", chunk)
    assert not supported_by_chunk("", chunk)


def test_supported_by_chunk_allows_small_paraphrase_with_token_recall():
    chunk = "Charles the Simple signed the treaty in 911."
    assert supported_by_chunk("Charles signed treaty 911", chunk, min_token_recall=0.8)
    assert not supported_by_chunk("Charles signed treaty 911", chunk, min_token_recall=1.01)
