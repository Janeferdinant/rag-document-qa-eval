from __future__ import annotations

import numpy as np
import pytest

from rag_qa_eval.reader import ReaderOutput, decide, decode_best_span, tune_threshold


def out(text: str, span: float, null: float) -> ReaderOutput:
    return ReaderOutput(text=text, start=0, end=len(text), span_score=span, null_score=null)


# ---- span decoding -------------------------------------------------------------------------


def test_decode_best_span_picks_the_highest_scoring_valid_pair():
    start = np.array([5.0, 0.1, 3.0, 0.0, 0.0])  # token 0 is the null/CLS position
    end = np.array([5.0, 0.0, 0.2, 4.0, 0.0])
    offsets = [(0, 0), (0, 3), (4, 7), (8, 12), (13, 17)]
    context = np.array([False, True, True, True, True])
    span = decode_best_span(start, end, offsets, context, max_answer_len=30)
    assert span == (4, 12, pytest.approx(7.0))  # tokens 2..3


def test_decode_best_span_never_returns_the_null_position_or_question_tokens():
    start = np.array([9.0, 9.0, 0.0, 0.0])
    end = np.array([9.0, 9.0, 0.0, 1.0])
    offsets = [(0, 0), (0, 2), (3, 5), (6, 9)]
    context = np.array([False, False, True, True])
    s, e, _ = decode_best_span(start, end, offsets, context, 30)  # type: ignore[misc]
    assert (s, e) == (3, 9) or (s, e) == (6, 9) or (s, e) == (3, 5)
    assert s >= 3


def test_decode_best_span_respects_the_maximum_answer_length_and_order():
    start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    end = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 10.0])
    start[1] = 10.0
    offsets = [(0, 0)] + [(i * 2, i * 2 + 1) for i in range(1, 6)]
    context = np.array([False] + [True] * 5)
    s, e, _ = decode_best_span(start, end, offsets, context, max_answer_len=3)  # type: ignore[misc]
    assert e - s <= 3 * 2  # at most three tokens wide


def test_decode_best_span_returns_none_without_context_tokens():
    z = np.zeros(3)
    assert decode_best_span(z, z, [(0, 0)] * 3, np.array([False] * 3), 30) is None


# ---- abstention logic ---------------------------------------------------------------------


def test_margin_is_span_minus_null_score():
    assert out("x", 7.0, 2.5).margin == pytest.approx(4.5)


def test_decide_answers_when_the_best_margin_clears_the_threshold():
    ans = decide([out("Paris", 8.0, 1.0)], threshold=0.0)
    assert ans.text == "Paris" and not ans.abstained and ans.chunk_index == 0


def test_decide_abstains_when_the_null_score_wins():
    ans = decide([out("Paris", 1.0, 3.0)], threshold=0.0)
    assert ans.abstained and ans.text is None and ans.chunk_index is None


def test_decide_picks_the_chunk_with_the_best_margin_not_the_best_span_score():
    outs = [out("A", 10.0, 9.0), out("B", 6.0, 1.0)]  # margins 1.0 and 5.0
    ans = decide(outs, threshold=0.0)
    assert ans.text == "B" and ans.chunk_index == 1


def test_decide_breaks_margin_ties_in_favour_of_the_higher_ranked_chunk():
    ans = decide([out("A", 5.0, 1.0), out("B", 5.0, 1.0)], threshold=0.0)
    assert ans.chunk_index == 0


def test_decide_threshold_is_inclusive_and_shifts_the_decision():
    outs = [out("A", 3.0, 1.0)]  # margin 2.0
    assert not decide(outs, threshold=2.0).abstained
    assert decide(outs, threshold=2.01).abstained
    assert not decide(outs, threshold=-5.0).abstained


def test_decide_abstains_on_empty_input_and_on_empty_span_text():
    assert decide([], threshold=0.0).abstained
    assert decide([out("", 9.0, 0.0)], threshold=0.0).abstained


# ---- threshold tuning ---------------------------------------------------------------------


def test_tune_threshold_finds_a_separating_threshold():
    # answerable questions have large margins, unanswerable ones negative margins
    margins = [5.0, 4.0, 3.0, -1.0, -2.0, -3.0]
    unanswerable = [False, False, False, True, True, True]
    t = tune_threshold(margins, unanswerable)
    assert -1.0 < t <= 3.0


def test_tune_threshold_requires_matching_nonempty_inputs():
    with pytest.raises(ValueError):
        tune_threshold([1.0], [True, False])
    with pytest.raises(ValueError):
        tune_threshold([], [])
