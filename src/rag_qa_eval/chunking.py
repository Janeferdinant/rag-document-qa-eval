"""Chunking strategies: fixed-size, sliding window and sentence-based.

All sizes are in whitespace-separated words. Every chunk keeps its character offsets into the
source document, so a chunk can be tested against a gold answer span without string matching.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

STRATEGIES = ("fixed", "sliding", "sentence")

_WORD = re.compile(r"\S+")
_BOUNDARY = re.compile(r"""[.!?]["')\]]*\s+|\n\s*\n""")
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "st", "vs", "etc", "e.g", "i.e", "no", "jr", "sr", "inc",
    "ltd", "co", "u.s", "u.k", "gen", "col", "lt", "capt", "mt", "fig", "vol", "approx", "dept",
    "est", "c", "ca", "cf", "al", "sgt", "rev", "hon", "gov", "sen", "rep", "pres",
}  # fmt: skip


@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    doc_id: int
    start: int  # character offset in the document (inclusive)
    end: int  # character offset in the document (exclusive)
    text: str


def _word_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _WORD.finditer(text)]


def _make(spans: list[tuple[int, int]], text: str, lo: int, hi: int) -> tuple[int, int, str]:
    start, end = spans[lo][0], spans[hi - 1][1]
    return start, end, text[start:end]


def _to_chunks(raw: list[tuple[int, int, str]], doc_id: int, first_chunk_id: int) -> list[Chunk]:
    return [Chunk(first_chunk_id + i, doc_id, s, e, t) for i, (s, e, t) in enumerate(raw)]


def _fixed_raw(text: str, size: int) -> list[tuple[int, int, str]]:
    if size < 1:
        raise ValueError("size must be >= 1")
    spans = _word_spans(text)
    return [_make(spans, text, i, min(i + size, len(spans))) for i in range(0, len(spans), size)]


def _sliding_raw(text: str, size: int, stride: int) -> list[tuple[int, int, str]]:
    if size < 1:
        raise ValueError("size must be >= 1")
    if not 1 <= stride <= size:
        raise ValueError("stride must be between 1 and size")
    spans = _word_spans(text)
    out: list[tuple[int, int, str]] = []
    i = 0
    while i < len(spans):
        hi = min(i + size, len(spans))
        out.append(_make(spans, text, i, hi))
        if hi == len(spans):
            break
        i += stride
    return out


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Rule-based sentence splitter returning (start, end) character spans.

    Splits after ., ! or ? followed by whitespace, and at blank lines, except after common
    abbreviations and single-letter initials.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _BOUNDARY.finditer(text):
        hard = m.group(0).strip() == ""  # blank-line boundary
        end = m.start() if hard else m.start() + len(m.group(0).rstrip())
        if not hard:
            prev = text[start : m.start() + 1].split()
            last = prev[-1].rstrip(".!?\"')]").lower() if prev else ""
            ends_with_period = text[m.start()] == "."
            if ends_with_period and (last in _ABBREVIATIONS or (len(last) == 1 and last.isalpha())):
                continue
        if text[start:end].strip():
            lead = len(text[start:end]) - len(text[start:end].lstrip())
            spans.append((start + lead, end))
        start = m.end()
    if text[start:].strip():
        lead = len(text[start:]) - len(text[start:].lstrip())
        spans.append((start + lead, len(text.rstrip())))
    return spans


def _sentence_raw(text: str, max_words: int, overlap_sentences: int = 0) -> list[tuple[int, int, str]]:
    if max_words < 1:
        raise ValueError("max_words must be >= 1")
    units: list[tuple[int, int]] = []  # sentences, hard-split when longer than max_words
    for s, e in split_sentences(text):
        piece = text[s:e]
        w = [(s + m.start(), s + m.end()) for m in _WORD.finditer(piece)]
        if len(w) <= max_words:
            units.append((s, e))
        else:
            for i in range(0, len(w), max_words):
                units.append((w[i][0], w[min(i + max_words, len(w)) - 1][1]))
    counts = [len(_WORD.findall(text[s:e])) for s, e in units]
    out: list[tuple[int, int, str]] = []
    i = 0
    while i < len(units):
        j, total = i, 0
        while j < len(units) and (total + counts[j] <= max_words or j == i):
            total += counts[j]
            j += 1
        out.append((units[i][0], units[j - 1][1], text[units[i][0] : units[j - 1][1]]))
        if j >= len(units):
            break
        i = max(i + 1, j - overlap_sentences)
    return out


def chunk_fixed(text: str, size: int) -> list[Chunk]:
    return _to_chunks(_fixed_raw(text, size), 0, 0)


def chunk_sliding(text: str, size: int, stride: int) -> list[Chunk]:
    return _to_chunks(_sliding_raw(text, size, stride), 0, 0)


def chunk_sentences(text: str, max_words: int, overlap_sentences: int = 0) -> list[Chunk]:
    return _to_chunks(_sentence_raw(text, max_words, overlap_sentences), 0, 0)


def chunk_document(
    doc_id: int,
    text: str,
    strategy: str,
    size: int,
    overlap: int | None = None,
    first_chunk_id: int = 0,
) -> list[Chunk]:
    """Chunk one document. `overlap` is the sliding-window overlap in words (default size // 2)."""
    if strategy == "fixed":
        raw = _fixed_raw(text, size)
    elif strategy == "sliding":
        ov = size // 2 if overlap is None else overlap
        raw = _sliding_raw(text, size, size - ov)
    elif strategy == "sentence":
        raw = _sentence_raw(text, size)
    else:
        raise ValueError(f"unknown chunking strategy: {strategy!r} (expected one of {STRATEGIES})")
    return _to_chunks(raw, doc_id, first_chunk_id)
