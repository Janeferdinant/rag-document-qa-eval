"""SQuAD 2.0 style answer metrics, abstention statistics and simple support checks."""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Sequence


def normalize_answer(s: str) -> str:
    """Lowercase, drop punctuation and the articles a/an/the, collapse whitespace (SQuAD rules)."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _f1_pair(prediction: str, gold: str) -> float:
    p, g = normalize_answer(prediction).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, golds: Sequence[str]) -> float:
    return max((float(normalize_answer(prediction) == normalize_answer(g)) for g in golds), default=0.0)


def f1_score(prediction: str, golds: Sequence[str]) -> float:
    return max((_f1_pair(prediction, g) for g in golds), default=0.0)


def score_prediction(prediction: str | None, golds: Sequence[str]) -> tuple[float, float]:
    """(EM, F1). `None` means the system abstained; an empty gold list means unanswerable."""
    if not golds:
        return (1.0, 1.0) if prediction is None else (0.0, 0.0)
    if prediction is None:
        return (0.0, 0.0)
    return exact_match(prediction, golds), f1_score(prediction, golds)


def abstention_stats(
    predicted_abstain: Sequence[bool], truly_unanswerable: Sequence[bool]
) -> tuple[float, float, float]:
    """Precision, recall and F1 of the 'abstain' decision (positive class = abstain)."""
    if len(predicted_abstain) != len(truly_unanswerable):
        raise ValueError("predictions and labels must have the same length")
    tp = sum(1 for a, u in zip(predicted_abstain, truly_unanswerable, strict=True) if a and u)
    n_pred, n_true = sum(predicted_abstain), sum(truly_unanswerable)
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_true if n_true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def answer_in_context(golds: Sequence[str], contexts: Sequence[str]) -> bool:
    """True if any gold answer occurs (after normalisation, on token boundaries) in the contexts."""
    haystack = " " + " ".join(normalize_answer(c) for c in contexts) + " "
    for g in golds:
        needle = normalize_answer(g)
        if needle and f" {needle} " in haystack:
            return True
    return False


def supported_by_chunk(answer: str, chunk: str, min_token_recall: float | None = None) -> bool:
    """Lexical support: the answer (normalised) is a token substring of the chunk. If
    `min_token_recall` is given, a paraphrase also counts when at least that fraction of the
    answer's tokens occur in the chunk. This is a cheap check for 'the cited chunk contains what
    the answer says'; it is not an entailment test."""
    a, c = normalize_answer(answer).split(), normalize_answer(chunk).split()
    if not a:
        return False
    if f" {' '.join(a)} " in f" {' '.join(c)} ":
        return True
    if min_token_recall is None:
        return False
    cset = set(c)
    return sum(1 for t in a if t in cset) / len(a) >= min_token_recall
