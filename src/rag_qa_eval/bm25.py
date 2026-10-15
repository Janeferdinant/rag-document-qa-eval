"""Okapi BM25 with an inverted index, plus a tokenizer shared with the browser demo.

The tokenizer and scoring are deliberately simple so that `docs/app.js` can reproduce them
exactly: lowercase, strip accents, split on [a-z0-9]+, drop a fixed stop-word list.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Any

import numpy as np

STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are as at be because been before
    being below between both but by can could did do does doing down during each few for from
    further had has have having he her here hers herself him himself his how i if in into is it
    its itself just me more most my myself no nor not of off on once only or other our ours
    ourselves out over own same she should so some such than that the their theirs them
    themselves then there these they this those through to too under until up very was we were
    what when where which while who whom why will with would you your yours yourself yourselves
    """.split()  # noqa: SIM905
)

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    folded = unicodedata.normalize("NFKD", text)
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch)).lower()
    tokens = _TOKEN.findall(folded)
    return [t for t in tokens if t not in STOPWORDS] if remove_stopwords else tokens


class BM25Index:
    """Okapi BM25 (Lucene-style idf) over pre-tokenised documents."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.doc_len = np.array([len(d) for d in docs], dtype=np.float64)
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0
        postings: dict[str, tuple[list[int], list[int]]] = {}
        for i, doc in enumerate(docs):
            for term, tf in Counter(doc).items():
                ids, tfs = postings.setdefault(term, ([], []))
                ids.append(i)
                tfs.append(tf)
        self.postings = {t: (np.array(i), np.array(f, dtype=np.float64)) for t, (i, f) in postings.items()}
        self.idf = {
            t: math.log(1 + (self.n - len(ids) + 0.5) / (len(ids) + 0.5))
            for t, (ids, _) in self.postings.items()
        }

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float64)
        if not self.n:
            return out
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / (self.avgdl or 1.0))
        for term in set(query):
            if term not in self.postings:
                continue
            ids, tfs = self.postings[term]
            out[ids] += self.idf[term] * tfs * (self.k1 + 1) / (tfs + norm[ids])
        return out

    def search(self, query: list[str], k: int) -> list[tuple[int, float]]:
        s = self.scores(query)
        if not s.size:
            return []
        k = min(k, s.size)
        top = np.argpartition(-s, k - 1)[:k] if k < s.size else np.arange(s.size)
        top = top[np.lexsort((top, -s[top]))]  # score desc, then id asc (deterministic)
        return [(int(i), float(s[i])) for i in top if s[i] > 0]

    def to_json(self) -> dict[str, Any]:
        return {
            "k1": self.k1,
            "b": self.b,
            "n": self.n,
            "avgdl": self.avgdl,
            "doc_len": self.doc_len.astype(int).tolist(),
            "postings": {
                t: [ids.tolist(), tfs.astype(int).tolist()] for t, (ids, tfs) in self.postings.items()
            },
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> BM25Index:
        obj = cls.__new__(cls)
        obj.k1, obj.b, obj.n, obj.avgdl = data["k1"], data["b"], data["n"], data["avgdl"]
        obj.doc_len = np.array(data["doc_len"], dtype=np.float64)
        obj.postings = {
            t: (np.array(ids), np.array(tfs, dtype=np.float64)) for t, (ids, tfs) in data["postings"].items()
        }
        obj.idf = {
            t: math.log(1 + (obj.n - len(ids) + 0.5) / (len(ids) + 0.5))
            for t, (ids, _) in obj.postings.items()
        }
        return obj
