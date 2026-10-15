"""Tiny fakes (templated corpus, hash encoder, rule-based reader): no model downloads, no network."""

from __future__ import annotations

import re
import zlib
from typing import Any

import numpy as np

from rag_qa_eval.bm25 import tokenize
from rag_qa_eval.evalcommon import Components, Settings
from rag_qa_eval.reader import ReaderOutput

N_ART, N_PARA = 8, 4


def make_squad() -> dict[str, Any]:
    arts = []
    for a in range(N_ART):
        paras = []
        for p in range(N_PARA):
            topic = f"Topic{a}x{p}"
            ctx = (
                f"{topic} was founded in the year {1800 + a * 10 + p} by Person{a}x{p}. "
                f"It is located in City{a}x{p} near the river. Many visitors come every summer "
                f"to see {topic} and enjoy the long walks along the water."
            )
            who = ctx.index(f"Person{a}x{p}")
            qas = [
                {
                    "id": f"a{a}p{p}q0",
                    "question": f"Who founded {topic}?",
                    "is_impossible": False,
                    "answers": [{"text": f"Person{a}x{p}", "answer_start": who}],
                },
                {
                    "id": f"a{a}p{p}q1",
                    "question": f"Who painted {topic}?",
                    "is_impossible": True,
                    "answers": [],
                },
            ]
            paras.append({"context": ctx, "qas": qas})
        arts.append({"title": f"Article{a}", "paragraphs": paras})
    return {"version": "v2.0", "data": arts}


class HashEncoder:
    model_name = "fake-hash"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(256, dtype=np.float32)
        for tok in tokenize(text):
            v[zlib.crc32(tok.encode()) % 256] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])


class OverlapReranker:
    def score(self, query: str, texts: list[str]) -> list[float]:
        q = set(tokenize(query))
        return [float(len(q & set(tokenize(t)))) for t in texts]


class FakeReader:
    """'Who founded X?' is answered from 'by PersonN' when the topic is in the passage."""

    def read_many(self, pairs: list[tuple[str, str]]) -> list[ReaderOutput]:
        outs = []
        for q, passage in pairs:
            topic = re.search(r"(Topic\w+)", q)
            m = re.search(r"by (Person\w+)", passage)
            has_topic = bool(topic and topic.group(1) in passage)
            if m and "founded" in q and has_topic:
                outs.append(ReaderOutput(m.group(1), m.start(1), m.end(1), 9.0, 1.0))
            elif m:
                outs.append(ReaderOutput(m.group(1), m.start(1), m.end(1), 1.0, 4.0))
            else:
                outs.append(ReaderOutput("", 0, 0, float("-inf"), 4.0))
        return outs


def components() -> Components:
    return Components(
        encoder=lambda name: HashEncoder(),
        reranker=lambda: OverlapReranker(),
        reader=lambda name: FakeReader(),
    )


def settings() -> Settings:
    return Settings(
        n_calib_articles=3,
        n_calib_questions=40,
        n_test_questions=60,
        strategies=("sliding", "sentence"),
        sizes=(20, 40),
        embedders=("fake-hash",),
        readers=("fake-reader",),
        reader_ks=(1, 3),
        n_boot=200,
        latency_queries=6,
    )
