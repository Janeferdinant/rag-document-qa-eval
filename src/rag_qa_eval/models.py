"""Thin wrappers around the openly licensed pretrained models (downloaded at run time)."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

# model id -> (licence, query prefix, source URL). Licences were read from the Hugging Face model
# cards on 2026-10-15; see THIRD_PARTY_NOTICES.md.
EMBEDDERS = {
    "sentence-transformers/all-MiniLM-L6-v2": ("Apache-2.0", ""),
    "BAAI/bge-small-en-v1.5": (
        "MIT",
        "Represent this sentence for searching relevant passages: ",
    ),
}
DEFAULT_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class SentenceEncoder:
    """Dense encoder (sentence-transformers). Embeddings are L2-normalised, so dot = cosine."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDER, batch_size: int = 64) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.query_prefix = EMBEDDERS.get(model_name, ("", ""))[1]
        self.model: Any = SentenceTransformer(model_name, device="cpu")

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                list(texts),
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

    def encode_queries(self, queries: Sequence[str]) -> np.ndarray:
        return self.encode([self.query_prefix + q for q in queries])


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER, batch_size: int = 32) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.batch_size = batch_size
        self.model: Any = CrossEncoder(model_name, device="cpu")

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        scores = self.model.predict(
            [(query, t) for t in texts], batch_size=self.batch_size, show_progress_bar=False
        )
        return [float(s) for s in scores]


def cached_embeddings(encoder: Any, texts: Sequence[str], cache_dir: Path) -> np.ndarray:
    """Encode `texts`, caching the result on disk keyed by model name and content hash."""
    h = hashlib.sha256(str(getattr(encoder, "model_name", type(encoder).__name__)).encode())
    for t in texts:
        h.update(b"\x00")
        h.update(t.encode())
    path = cache_dir / f"emb-{h.hexdigest()[:24]}.npy"
    if path.exists():
        return np.load(path)
    emb = encoder.encode(texts)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    return emb
