from __future__ import annotations

import numpy as np
import pytest

from rag_qa_eval.bm25 import tokenize
from rag_qa_eval.chunking import Chunk
from rag_qa_eval.retrieval import METHODS, Retriever, chunk_contains_span, relevant_chunk_ids

TEXTS = [
    "The Normans were the people who gave their name to Normandy in France.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "The Amazon rainforest covers much of northwestern Brazil.",
    "Alan Turing proposed the imitation game as a test of machine intelligence.",
]
VOCAB = sorted({t for s in TEXTS for t in tokenize(s)} | {"normans", "turing", "plants"})


class BowEncoder:
    """Bag-of-words embeddings: deterministic, no model download."""

    def encode(self, texts: list[str]) -> np.ndarray:
        m = np.zeros((len(texts), len(VOCAB)), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                if tok in VOCAB:
                    m[i, VOCAB.index(tok)] += 1.0
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.where(norms == 0, 1, norms)


class LengthReranker:
    """Prefers shorter passages: lets tests check that reranking reorders the candidates."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [-float(len(t)) for t in texts]


def chunks() -> list[Chunk]:
    out, pos = [], 0
    for i, t in enumerate(TEXTS):
        out.append(Chunk(i, 0, pos, pos + len(t), t))
        pos += len(t) + 2
    return out


def retriever(**kw) -> Retriever:
    return Retriever(chunks(), encoder=BowEncoder(), reranker=LengthReranker(), **kw)


def test_methods_are_the_documented_four():
    assert METHODS == ("bm25", "dense", "hybrid", "hybrid+rerank")


@pytest.mark.parametrize("method", ["bm25", "dense", "hybrid"])
def test_each_method_finds_the_obvious_chunk(method: str):
    r = retriever()
    assert r.search("Who proposed the imitation game?", method, k=3)[0] == 3
    assert r.search("Where is the Amazon rainforest?", method, k=3)[0] == 2


def test_hybrid_plus_rerank_applies_the_reranker_to_the_candidates():
    r = retriever(rerank_depth=4)
    ranked = r.search("plants energy sunlight Normans", "hybrid+rerank", k=4)
    lengths = [len(TEXTS[i]) for i in ranked]
    assert lengths == sorted(lengths)  # reranker prefers short passages


def test_rerank_keeps_candidates_beyond_the_rerank_depth_in_fused_order():
    r = retriever(rerank_depth=2)
    fused = r.search("Normans Normandy France", "hybrid", k=4)
    reranked = r.search("Normans Normandy France", "hybrid+rerank", k=4)
    assert reranked[2:] == fused[2:]
    assert set(reranked[:2]) == set(fused[:2])


def test_search_batch_matches_single_queries():
    r = retriever()
    qs = ["imitation game", "Amazon Brazil", "sunlight plants"]
    batch = r.search_batch(qs, "hybrid", k=3)
    assert batch == [r.search(q, "hybrid", k=3) for q in qs]


def test_k_larger_than_the_corpus_is_capped_and_unknown_method_rejected():
    r = retriever()
    assert len(r.search("normans", "dense", k=50)) == 4
    with pytest.raises(ValueError):
        r.search("x", "magic", k=3)


def test_dense_and_rerank_require_their_components():
    with pytest.raises(ValueError):
        Retriever(chunks()).search("x", "dense", k=1)
    with pytest.raises(ValueError):
        Retriever(chunks(), encoder=BowEncoder()).search("x", "hybrid+rerank", k=1)


def test_timed_search_reports_each_stage_in_milliseconds():
    ids, timings = retriever().search_timed("imitation game", "hybrid+rerank", k=3)
    assert len(ids) == 3
    assert set(timings) >= {"bm25_ms", "encode_ms", "dense_ms", "fuse_ms", "rerank_ms"}
    assert all(v >= 0 for v in timings.values())


def test_precomputed_embeddings_are_used_instead_of_re_encoding():
    enc = BowEncoder()
    emb = enc.encode(TEXTS)

    class Boom:
        def encode(self, texts: list[str]) -> np.ndarray:
            if len(texts) == len(TEXTS):
                raise AssertionError("corpus must not be re-encoded")
            return enc.encode(texts)

    r = Retriever(chunks(), encoder=Boom(), embeddings=emb)
    assert r.search("imitation game", "dense", k=1) == [3]


# ---- relevance against gold spans ------------------------------------------------------------


def test_chunk_contains_span_requires_the_whole_span_in_the_same_document():
    c = Chunk(0, 1, 100, 200, "x")
    assert chunk_contains_span(c, 1, [(120, 130)])
    assert not chunk_contains_span(c, 1, [(95, 130)])  # cut by the chunk boundary
    assert not chunk_contains_span(c, 2, [(120, 130)])  # other document
    assert not chunk_contains_span(c, 1, [])  # unanswerable question


def test_relevant_chunk_ids_lists_all_overlapping_windows():
    cs = [Chunk(0, 0, 0, 50, ""), Chunk(1, 0, 25, 75, ""), Chunk(2, 0, 50, 100, "")]
    assert relevant_chunk_ids(cs, 0, [(30, 40)]) == {0, 1}
    assert relevant_chunk_ids(cs, 0, [(60, 70)]) == {1, 2}
