"""Extractive reader with abstention.

A SQuAD 2.0 span-extraction model scores every candidate span and also a 'null' answer (the
[CLS] position). The difference `span_score - null_score` is the *margin*. For each retrieved
chunk we keep the best span and the margin; the answer comes from the chunk with the largest
margin, and the system abstains when that margin is below a threshold tuned on held-out
calibration questions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

DEFAULT_READER = "deepset/minilm-uncased-squad2"


@dataclass(frozen=True)
class ReaderOutput:
    text: str  # best non-null span ('' if the passage has no usable span)
    start: int  # character offsets of the span inside the passage
    end: int
    span_score: float  # start_logit + end_logit of the best span
    null_score: float  # start_logit + end_logit at the null position

    @property
    def margin(self) -> float:
        return self.span_score - self.null_score


@dataclass(frozen=True)
class Answer:
    text: str | None  # None means the system abstained
    chunk_index: int | None  # position of the cited chunk in the retrieved list
    margin: float
    start: int = 0
    end: int = 0

    @property
    def abstained(self) -> bool:
        return self.text is None


def decode_best_span(
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    offsets: Sequence[tuple[int, int]],
    context_mask: np.ndarray,
    max_answer_len: int = 30,
) -> tuple[int, int, float] | None:
    """Best (char_start, char_end, score) over context tokens; end - start < max_answer_len."""
    n = len(start_logits)
    idx = np.flatnonzero(context_mask)
    if idx.size == 0:
        return None
    scores = start_logits[:, None] + end_logits[None, :]
    rows = np.arange(n)[:, None]
    cols = np.arange(n)[None, :]
    valid = (cols >= rows) & (cols - rows < max_answer_len)
    valid &= context_mask[:, None] & context_mask[None, :]
    if not valid.any():
        return None
    scores = np.where(valid, scores, -np.inf)
    s, e = np.unravel_index(int(np.argmax(scores)), scores.shape)
    return offsets[int(s)][0], offsets[int(e)][1], float(scores[s, e])


def decide(outputs: Sequence[ReaderOutput], threshold: float = 0.0) -> Answer:
    """Pick the chunk with the largest margin; abstain if that margin is below `threshold`."""
    best_i, best = -1, None
    for i, o in enumerate(outputs):
        if o.text and (best is None or o.margin > best.margin):
            best_i, best = i, o
    if best is None:
        return Answer(None, None, float("-inf"))
    if best.margin < threshold:
        return Answer(None, None, best.margin)
    return Answer(best.text, best_i, best.margin, best.start, best.end)


def tune_threshold(
    margins: Sequence[float],
    unanswerable: Sequence[bool],
    answered_score: Sequence[float] | None = None,
) -> float:
    """Pick the abstention threshold that maximises the mean per-question score.

    Abstaining scores 1 on an unanswerable question and 0 on an answerable one. Answering scores
    `answered_score[i]` (e.g. F1) on an answerable question and 0 on an unanswerable one; by
    default `answered_score` is 1 for every answerable question (pure answerable-vs-not accuracy).
    Candidates are midpoints between sorted distinct margins plus the two extremes.
    """
    if len(margins) != len(unanswerable) or not len(margins):
        raise ValueError("margins and labels must be non-empty and of equal length")
    m = np.asarray(margins, dtype=np.float64)
    u = np.asarray(unanswerable, dtype=bool)
    ans = np.ones(len(m)) if answered_score is None else np.asarray(answered_score, dtype=np.float64)
    ans = np.where(u, 0.0, ans)
    abst = u.astype(np.float64)
    finite = m[np.isfinite(m)]
    distinct = np.unique(finite)
    cands = [float(distinct[0]) - 1.0] if distinct.size else [0.0]
    cands += [float((a + b) / 2) for a, b in zip(distinct[:-1], distinct[1:], strict=False)]
    cands += [float(distinct[-1]) + 1.0] if distinct.size else []
    best_t, best_v = 0.0, -1.0
    for t in cands:
        v = float(np.where(m < t, abst, ans).mean())
        if v > best_v + 1e-12:
            best_t, best_v = t, v
    return best_t


class ExtractiveReader:
    """Hugging Face extractive QA model (default: deepset/minilm-uncased-squad2, CC BY 4.0)."""

    def __init__(
        self,
        model_name: str = DEFAULT_READER,
        max_length: int = 384,
        stride: int = 128,
        max_answer_len: int = 30,
        batch_size: int = 16,
    ) -> None:
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        self.model_name = model_name
        self.max_length, self.stride = max_length, stride
        self.max_answer_len, self.batch_size = max_answer_len, batch_size
        self._torch = torch
        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
        self.model: Any = AutoModelForQuestionAnswering.from_pretrained(model_name).eval()

    def read_many(self, pairs: Sequence[tuple[str, str]]) -> list[ReaderOutput]:
        """Read (question, passage) pairs. Long passages are split into overlapping windows; the
        best span over windows is kept and the null score is the minimum over windows."""
        results: list[ReaderOutput] = []
        for lo in range(0, len(pairs), self.batch_size):
            results.extend(self._read_batch(pairs[lo : lo + self.batch_size]))
        return results

    def read(self, question: str, passage: str) -> ReaderOutput:
        return self.read_many([(question, passage)])[0]

    def _read_batch(self, pairs: Sequence[tuple[str, str]]) -> list[ReaderOutput]:
        torch = self._torch
        enc = self.tokenizer(
            [q for q, _ in pairs],
            [p for _, p in pairs],
            truncation="only_second",
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=True,
            return_tensors="pt",
        )
        owner = enc["overflow_to_sample_mapping"].tolist()
        offsets = enc.pop("offset_mapping").tolist()
        enc.pop("overflow_to_sample_mapping")
        with torch.no_grad():
            logits = self.model(**enc)
        start_all = logits.start_logits.numpy()
        end_all = logits.end_logits.numpy()
        best: dict[int, tuple[int, int, float]] = {}
        null: dict[int, float] = {}
        for w, sample in enumerate(owner):
            seq_ids = enc.sequence_ids(w)
            mask = np.array([s == 1 for s in seq_ids])
            span = decode_best_span(start_all[w], end_all[w], offsets[w], mask, self.max_answer_len)
            n_score = float(start_all[w][0] + end_all[w][0])
            null[sample] = min(null.get(sample, n_score), n_score)
            if span is not None and (sample not in best or span[2] > best[sample][2]):
                best[sample] = span
        out: list[ReaderOutput] = []
        for i, (_, passage) in enumerate(pairs):
            if i in best:
                s, e, score = best[i]
                out.append(ReaderOutput(passage[s:e], s, e, score, null[i]))
            else:
                out.append(ReaderOutput("", 0, 0, float("-inf"), null.get(i, 0.0)))
        return out
