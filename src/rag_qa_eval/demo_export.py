"""Export the static demo's data: a 3-article sample corpus, its BM25 index and precomputed examples.

The precomputed examples are produced by re-running the committed evaluation's headline pipeline
on questions from the sample articles, and each one is checked against the committed
per-question results of the reader run (same outcome category), so the page only shows outcomes
that the evaluation recorded.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .analysis import anonymise_id
from .bm25 import STOPWORDS, BM25Index, tokenize
from .data import Document
from .eval_reader import build_conditions, evaluate_condition, read_all
from .eval_retrieval import make_retriever
from .evalcommon import Settings, chunk_corpus, default_components, latest_run, prepare
from .reader import ReaderOutput

OUTCOME_LABELS = {
    "correct": "Correct: exact match with a gold answer.",
    "retrieval_failure": "Retrieval failure: none of the retrieved passages contains a gold answer.",
    "reading_failure_abstained": "Reading failure: the answer was in the retrieved passages but the reader abstained.",
    "reading_failure_wrong_span": "Reading failure: the answer was in the retrieved passages but the reader chose a different span.",
    "boundary_mismatch": "Partial match: the reader's span overlaps the gold answer but is not an exact match.",
    "correct_abstention": "Correct abstention: the question is unanswerable and the reader abstained.",
    "false_answer": "False answer: the question is unanswerable but the reader gave an answer.",
}
QUOTA = (
    ("correct", 2),
    ("retrieval_failure", 1),
    ("reading", 1),
    ("correct_abstention", 1),
    ("false_answer", 1),
)
READING = ("reading_failure_abstained", "reading_failure_wrong_span", "boundary_mismatch")
N_DEMO_ARTICLES = 3


def settings_from_dict(d: dict[str, Any]) -> Settings:
    fixed: dict[str, Any] = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()}
    return Settings(**fixed)


def export_demo(data_dir: Path, runs_dir: Path, out_dir: Path) -> None:
    reader_run = latest_run(runs_dir, "reader")
    if reader_run is None:
        raise FileNotFoundError("run the reader stage first")
    settings = settings_from_dict(reader_run["meta"]["settings"])
    ch, head = reader_run["meta"]["chunking"], reader_run["headline"]
    prep = prepare(settings, data_dir)

    rng = random.Random(settings.seed)
    titles = sorted(rng.sample(sorted(prep.test_titles), N_DEMO_ARTICLES))
    demo_docs = [
        Document(i, prep.doc_by_title[t].title, prep.doc_by_title[t].text, prep.doc_by_title[t].para_starts)
        for i, t in enumerate(titles)
    ]
    chunks = chunk_corpus(demo_docs, ch["strategy"], ch["size"])
    index = BM25Index([tokenize(c.text) for c in chunks])
    attribution = (
        "Passages are excerpts of Wikipedia articles as distributed in SQuAD 2.0 (Rajpurkar, Jia and Liang, 2018), "
        "licensed CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/), shown unmodified. "
        f"Sample articles: {', '.join(t.replace('_', ' ') for t in titles)}. "
        "Precomputed examples may show passages from other SQuAD 2.0 dev articles."
    )
    corpus = {
        "meta": {
            "attribution": attribution,
            "articles": titles,
            "chunking": {"strategy": ch["strategy"], "size": ch["size"]},
            "retrieval": "BM25 (k1=1.5, b=0.75), top 3",
        },
        "chunks": [
            {"id": i, "title": demo_docs[c.doc_id].title, "text": c.text} for i, c in enumerate(chunks)
        ],
    }

    # Precomputed examples: the headline pipeline over the full corpus, restricted to sample articles.
    comps = default_components()
    full_chunks = chunk_corpus(prep.docs, ch["strategy"], ch["size"])
    retriever = make_retriever(comps, full_chunks, ch["embedder"], True, data_dir / "cache", settings.depth)
    reader = comps.get_reader(head["reader"])
    test_qs = [
        q for q in prep.sample("test", settings.n_test_questions, settings.seed, False) if q.title in titles
    ]
    ranked = retriever.search_batch([q.text for q in test_qs], head["method"], head["k"])
    pairs = {}
    for q, r in zip(test_qs, ranked, strict=True):
        for p in r:
            pairs[(q.qid, p)] = (q.text, full_chunks[p].text)
    cache: dict[tuple[str, int], ReaderOutput] = {}
    read_all(reader, pairs, cache, lambda _m: None)
    cond = build_conditions(prep, test_qs, full_chunks, ranked, head["k"], cache)
    _, rows, _ = evaluate_condition(cond, head["threshold"], 50, settings.seed)
    committed = {r["qid"]: r for r in reader_run["per_question"]}
    by_cat: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        ref = committed.get(anonymise_id(row["qid"]))
        if ref is None or ref["category"] != row["category"] or ref["abstained"] != row["abstained"]:
            continue  # only keep outcomes that match the committed evaluation exactly
        by_cat.setdefault(row["category"], []).append(i)
    chosen: list[int] = []
    for cat, n in QUOTA:
        pool = sorted(
            by_cat.get(cat, []) if cat != "reading" else [i for c in READING for i in by_cat.get(c, [])]
        )
        chosen += rng.sample(pool, min(n, len(pool)))
    examples = []
    for i in chosen:
        q, row = test_qs[i], rows[i]
        examples.append(
            {
                "id": anonymise_id(q.qid),
                "question": q.text,
                "gold": sorted({a.text for a in q.answers})[:3],
                "answer": row["prediction"],
                "cited_rank": None if row["cited_rank"] is None else row["cited_rank"] + 1,
                "category": row["category"],
                "outcome_label": OUTCOME_LABELS[row["category"]],
                "retrieved": [
                    {
                        "rank": r + 1,
                        "title": prep.docs[full_chunks[p].doc_id].title,
                        "text": full_chunks[p].text,
                    }
                    for r, p in enumerate(ranked[i])
                ],
            }
        )
    pipeline = (
        f"extractive reader {head['reader']}, {head['method']} retrieval with {ch['embedder'].split('/')[-1]}, "
        f"{ch['strategy']} chunks of {ch['size']} words, top {head['k']} passages, "
        f"abstention margin threshold {head['threshold']}"
    )
    ex = {
        "meta": {
            "date": reader_run["meta"]["date"],
            "pipeline": pipeline,
            "selection_rule": "seeded pick per outcome category among test questions from the sample articles",
        },
        "examples": examples,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sample_corpus.json").write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf8")
    (out_dir / "bm25_index.json").write_text(
        json.dumps({"stopwords": sorted(STOPWORDS), "index": index.to_json()}), encoding="utf8"
    )
    (out_dir / "examples.json").write_text(json.dumps(ex, ensure_ascii=False, indent=1), encoding="utf8")
