from __future__ import annotations

import math

import numpy as np

from rag_qa_eval.bm25 import STOPWORDS, BM25Index, tokenize


def test_tokenize_lowercases_strips_accents_and_stopwords():
    assert tokenize("The Café of São Paulo, 1999!") == ["cafe", "sao", "paulo", "1999"]


def test_tokenize_can_keep_stopwords():
    assert tokenize("The cat", remove_stopwords=False) == ["the", "cat"]
    assert "the" in STOPWORDS


def test_bm25_matches_a_hand_computed_score():
    docs = [["apple", "banana"], ["apple", "apple", "cherry"], ["durian"]]
    index = BM25Index(docs, k1=1.5, b=0.75)
    n, df = 3, 2  # "apple" occurs in 2 of 3 documents
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    avgdl = (2 + 3 + 1) / 3

    def part(tf: int, dl: int) -> float:
        return idf * tf * 2.5 / (tf + 1.5 * (1 - 0.75 + 0.75 * dl / avgdl))

    scores = index.scores(["apple"])
    assert np.allclose(scores, [part(1, 2), part(2, 3), 0.0])


def test_bm25_ranks_the_matching_document_first_and_ignores_unknown_terms():
    docs = [["red", "fox"], ["blue", "whale"], ["green", "tree"]]
    index = BM25Index(docs)
    assert index.search(["whale", "zzz"], 2)[0][0] == 1
    assert np.all(index.scores(["unseen"]) == 0.0)


def test_bm25_search_returns_only_positive_scores_in_descending_order():
    index = BM25Index([["a", "b"], ["b", "c"], ["d"]])
    hits = index.search(["b"], 5)
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(s > 0 for s in scores)
    assert len(hits) == 2


def test_bm25_empty_corpus_and_empty_query_are_safe():
    assert BM25Index([]).scores(["x"]).shape == (0,)
    index = BM25Index([["a"]])
    assert index.search([], 3) == []


def test_bm25_json_roundtrip_preserves_scores():
    docs = [["alpha", "beta"], ["beta", "gamma", "gamma"], ["delta"]]
    index = BM25Index(docs)
    clone = BM25Index.from_json(index.to_json())
    assert np.allclose(index.scores(["gamma", "beta"]), clone.scores(["gamma", "beta"]))
