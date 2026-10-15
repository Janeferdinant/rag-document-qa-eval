"""Retrieval over chunks: BM25, dense, hybrid (reciprocal rank fusion) and an optional reranker."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from .bm25 import BM25Index, tokenize
from .chunking import Chunk
from .fusion import reciprocal_rank_fusion

METHODS = ("bm25", "dense", "hybrid", "hybrid+rerank")


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


def chunk_contains_span(chunk: Chunk, doc_id: int, spans: Sequence[tuple[int, int]]) -> bool:
    """True if the chunk is in the document and fully contains at least one gold answer span."""
    return chunk.doc_id == doc_id and any(chunk.start <= s and e <= chunk.end for s, e in spans)


def relevant_chunk_ids(chunks: Sequence[Chunk], doc_id: int, spans: Sequence[tuple[int, int]]) -> set[int]:
    """Positions (indices into `chunks`) of all chunks that fully contain a gold answer span."""
    return {i for i, c in enumerate(chunks) if chunk_contains_span(c, doc_id, spans)}


def _top_k(scores: np.ndarray, k: int) -> list[int]:
    k = min(k, scores.size)
    if k <= 0:
        return []
    top = np.argpartition(-scores, k - 1)[:k] if k < scores.size else np.arange(scores.size)
    return [int(i) for i in top[np.lexsort((top, -scores[top]))]]


class Retriever:
    """Ranks chunks for a query. `search` returns positions in the `chunks` list."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        encoder: Encoder | None = None,
        reranker: Reranker | None = None,
        embeddings: np.ndarray | None = None,
        rrf_k: int = 60,
        candidate_depth: int = 100,
        rerank_depth: int = 20,
    ) -> None:
        self.chunks = list(chunks)
        self.encoder, self.reranker = encoder, reranker
        self.rrf_k, self.candidate_depth, self.rerank_depth = rrf_k, candidate_depth, rerank_depth
        self.bm25 = BM25Index([tokenize(c.text) for c in self.chunks])
        if embeddings is None and encoder is not None:
            embeddings = encoder.encode([c.text for c in self.chunks])
        self.embeddings = embeddings

    # -- stages ---------------------------------------------------------------------------
    def _encode_queries(self, queries: list[str]) -> np.ndarray:
        enc: Any = self.encoder
        fn = getattr(enc, "encode_queries", enc.encode)
        return np.asarray(fn(queries), dtype=np.float32)

    def _require(self, method: str) -> None:
        if method not in METHODS:
            raise ValueError(f"unknown retrieval method {method!r}; expected one of {METHODS}")
        if method != "bm25" and (self.encoder is None or self.embeddings is None):
            raise ValueError(f"method {method!r} needs an encoder")
        if method == "hybrid+rerank" and self.reranker is None:
            raise ValueError("method 'hybrid+rerank' needs a reranker")

    def _dense_rank(self, qvec: np.ndarray, depth: int) -> list[int]:
        assert self.embeddings is not None
        return _top_k(self.embeddings @ qvec, depth)

    def _bm25_rank(self, query: str, depth: int) -> list[int]:
        return [i for i, _ in self.bm25.search(tokenize(query), depth)]

    def _rerank(self, query: str, fused: list[int], k: int) -> list[int]:
        assert self.reranker is not None
        head, tail = fused[: self.rerank_depth], fused[self.rerank_depth :]
        scores = self.reranker.score(query, [self.chunks[i].text for i in head])
        order = sorted(range(len(head)), key=lambda j: (-scores[j], j))
        return ([head[j] for j in order] + tail)[:k]

    # -- public API -----------------------------------------------------------------------
    def search_timed(self, query: str, method: str, k: int) -> tuple[list[int], dict[str, float]]:
        self._require(method)
        t: dict[str, float] = {}
        depth = max(self.candidate_depth, k)
        clock = time.perf_counter

        def lap(name: str, t0: float) -> None:
            t[name] = (clock() - t0) * 1000.0

        if method == "bm25":
            t0 = clock()
            ids = self._bm25_rank(query, k)
            lap("bm25_ms", t0)
            return ids, t
        qvec = None
        t0 = clock()
        qvec = self._encode_queries([query])[0]
        lap("encode_ms", t0)
        t0 = clock()
        dense = self._dense_rank(qvec, depth)
        lap("dense_ms", t0)
        if method == "dense":
            return dense[:k], t
        t0 = clock()
        sparse = self._bm25_rank(query, depth)
        lap("bm25_ms", t0)
        t0 = clock()
        fused = [i for i, _ in reciprocal_rank_fusion([sparse, dense], k=self.rrf_k)]
        lap("fuse_ms", t0)
        if method == "hybrid":
            return fused[:k], t
        t0 = clock()
        ids = self._rerank(query, fused, k)
        lap("rerank_ms", t0)
        return ids, t

    def search(self, query: str, method: str, k: int) -> list[int]:
        return self.search_timed(query, method, k)[0]

    def search_batch(self, queries: list[str], method: str, k: int) -> list[list[int]]:
        """Same result as calling `search` per query, but encodes all queries in one batch."""
        self._require(method)
        if method == "bm25":
            return [self._bm25_rank(q, k) for q in queries]
        depth = max(self.candidate_depth, k)
        qvecs = self._encode_queries(queries)
        out: list[list[int]] = []
        for q, v in zip(queries, qvecs, strict=True):
            dense = self._dense_rank(v, depth)
            if method == "dense":
                out.append(dense[:k])
                continue
            sparse = self._bm25_rank(q, depth)
            fused = [i for i, _ in reciprocal_rank_fusion([sparse, dense], k=self.rrf_k)]
            out.append(fused[:k] if method == "hybrid" else self._rerank(q, fused, k))
        return out
