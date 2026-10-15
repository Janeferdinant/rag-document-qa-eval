from __future__ import annotations

import pytest

from rag_qa_eval.analysis import (
    CATEGORIES,
    anonymise_id,
    categorize,
    category_counts,
    pick_examples,
    question_metrics,
)


def test_unanswerable_questions_are_correct_abstentions_or_false_answers():
    assert categorize(answerable=False, abstained=True, in_context=False, em=1.0, f1=1.0) == (
        "correct_abstention"
    )
    assert categorize(answerable=False, abstained=False, in_context=False, em=0.0, f1=0.0) == ("false_answer")


def test_answerable_correct_answer_is_correct_even_without_a_gold_string_in_context():
    assert categorize(answerable=True, abstained=False, in_context=True, em=1.0, f1=1.0) == ("correct")


def test_answerable_without_the_answer_in_the_retrieved_chunks_is_a_retrieval_failure():
    assert categorize(answerable=True, abstained=False, in_context=False, em=0.0, f1=0.0) == (
        "retrieval_failure"
    )
    assert categorize(answerable=True, abstained=True, in_context=False, em=0.0, f1=0.0) == (
        "retrieval_failure"
    )


def test_answerable_with_the_answer_in_context_splits_the_reading_failures():
    assert categorize(answerable=True, abstained=True, in_context=True, em=0.0, f1=0.0) == (
        "reading_failure_abstained"
    )
    assert categorize(answerable=True, abstained=False, in_context=True, em=0.0, f1=0.0) == (
        "reading_failure_wrong_span"
    )
    assert categorize(answerable=True, abstained=False, in_context=True, em=0.0, f1=0.6) == (
        "boundary_mismatch"
    )


def test_every_category_returned_is_documented():
    assert "correct" in CATEGORIES and "retrieval_failure" in CATEGORIES


def test_category_counts_include_zero_categories_and_sum_to_the_total():
    counts = category_counts(["correct", "correct", "false_answer"])
    assert counts["correct"] == 2 and counts["retrieval_failure"] == 0
    assert sum(counts.values()) == 3
    with pytest.raises(ValueError):
        category_counts(["nonsense"])


def test_question_metrics_scores_abstention_and_spans():
    assert question_metrics(None, []) == {"em": 1.0, "f1": 1.0}
    assert question_metrics("Paris", ["Paris"]) == {"em": 1.0, "f1": 1.0}
    assert question_metrics("Berlin", ["Paris"]) == {"em": 0.0, "f1": 0.0}


def test_anonymise_id_is_stable_short_and_hides_the_original():
    a = anonymise_id("56ddde6b9a695914005b9628")
    assert a == anonymise_id("56ddde6b9a695914005b9628") and len(a) == 8
    assert "56ddde" not in a


def test_pick_examples_is_seeded_and_capped_per_category():
    rows = [{"qid": f"q{i}", "category": "correct" if i % 2 else "false_answer"} for i in range(20)]
    a = pick_examples(rows, per_category=3, seed=1)
    b = pick_examples(rows, per_category=3, seed=1)
    assert a == b
    assert sum(r["category"] == "correct" for r in a) == 3
    assert sum(r["category"] == "false_answer" for r in a) == 3
