"""SQuAD 2.0 dev set: download with checksum, parse, build documents, split and sample.

SQuAD 2.0 (Rajpurkar, Jia and Liang, 2018) is distributed under CC BY-SA 4.0. The raw file is
downloaded at run time into `data/raw/` (git-ignored) and is never committed.
"""

from __future__ import annotations

import hashlib
import json
import random
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SQUAD_DEV_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
SQUAD_DEV_SHA256 = "80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8"
SQUAD_LICENSE = "CC BY-SA 4.0"


@dataclass(frozen=True)
class Answer:
    text: str
    start: int  # character offset inside the paragraph


@dataclass(frozen=True)
class Question:
    qid: str
    text: str
    title: str
    para_idx: int
    answers: tuple[Answer, ...]
    is_impossible: bool


@dataclass(frozen=True)
class Article:
    title: str
    paragraphs: tuple[str, ...]


@dataclass(frozen=True)
class Document:
    doc_id: int
    title: str
    text: str
    para_starts: list[int]  # offset of each paragraph inside `text`


@dataclass(frozen=True)
class SquadData:
    articles: list[Article]
    questions: list[Question]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download_file(url: str, dest: Path, sha256: str, timeout: int = 120) -> Path:
    """Download `url` to `dest` unless a file with the expected checksum is already there."""
    if dest.exists() and sha256_file(dest) == sha256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "rag-document-qa-eval/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as out:
        for block in iter(lambda: resp.read(1 << 20), b""):
            out.write(block)
    if sha256_file(tmp) != sha256:
        tmp.unlink()
        raise ValueError(f"checksum mismatch for {url}: expected {sha256}")
    tmp.replace(dest)
    return dest


def parse_squad(obj: dict[str, Any]) -> SquadData:
    articles: list[Article] = []
    questions: list[Question] = []
    for art in obj["data"]:
        title = art["title"]
        paragraphs: list[str] = []
        for p_idx, para in enumerate(art["paragraphs"]):
            paragraphs.append(para["context"])
            for qa in para["qas"]:
                answers = tuple(Answer(a["text"], a["answer_start"]) for a in qa.get("answers", []))
                questions.append(
                    Question(
                        qid=qa["id"],
                        text=qa["question"],
                        title=title,
                        para_idx=p_idx,
                        answers=answers,
                        is_impossible=bool(qa.get("is_impossible", False)) or not answers,
                    )
                )
        articles.append(Article(title, tuple(paragraphs)))
    return SquadData(articles, questions)


def load_squad(path: Path) -> SquadData:
    return parse_squad(json.loads(path.read_text(encoding="utf-8")))


def build_documents(articles: Sequence[Article]) -> list[Document]:
    """One document per article: paragraphs joined by a blank line, with paragraph offsets."""
    docs: list[Document] = []
    for i, art in enumerate(articles):
        starts: list[int] = []
        pos = 0
        for para in art.paragraphs:
            starts.append(pos)
            pos += len(para) + 2
        docs.append(Document(i, art.title, "\n\n".join(art.paragraphs), starts))
    return docs


def gold_spans(question: Question, doc: Document) -> list[tuple[int, int]]:
    """Gold answer spans as (start, end) character offsets in the document text."""
    base = doc.para_starts[question.para_idx]
    return [(base + a.start, base + a.start + len(a.text)) for a in question.answers]


def split_titles(titles: Sequence[str], seed: int, n_calibration: int) -> tuple[list[str], list[str]]:
    """Seeded article-level split into (calibration, test). Questions never share an article."""
    shuffled = sorted(titles)
    random.Random(seed).shuffle(shuffled)
    return sorted(shuffled[:n_calibration]), sorted(shuffled[n_calibration:])


def sample_questions(
    questions: Sequence[Question],
    titles: Sequence[str],
    n: int,
    seed: int,
    answerable_only: bool = False,
) -> list[Question]:
    """Seeded sample (without replacement, sorted by id) of questions from the given articles."""
    allowed = set(titles)
    pool = sorted(
        (q for q in questions if q.title in allowed and not (answerable_only and q.is_impossible)),
        key=lambda q: q.qid,
    )
    if n >= len(pool):
        return pool
    return sorted(random.Random(seed).sample(pool, n), key=lambda q: q.qid)
