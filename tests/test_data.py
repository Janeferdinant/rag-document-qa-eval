from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rag_qa_eval.data import (
    build_documents,
    download_file,
    gold_spans,
    parse_squad,
    sample_questions,
    sha256_file,
    split_titles,
)

SQUAD = {
    "version": "v2.0",
    "data": [
        {
            "title": "Normans",
            "paragraphs": [
                {
                    "context": "The Normans were in Normandy. They came from Denmark.",
                    "qas": [
                        {
                            "id": "q1",
                            "question": "Where were the Normans?",
                            "is_impossible": False,
                            "answers": [{"text": "Normandy", "answer_start": 20}],
                        },
                        {
                            "id": "q2",
                            "question": "Who ruled Spain?",
                            "is_impossible": True,
                            "answers": [],
                            "plausible_answers": [{"text": "Denmark", "answer_start": 44}],
                        },
                    ],
                },
                {
                    "context": "Rollo was a leader.",
                    "qas": [
                        {
                            "id": "q3",
                            "question": "What was Rollo?",
                            "is_impossible": False,
                            "answers": [{"text": "a leader", "answer_start": 10}],
                        }
                    ],
                },
            ],
        },
        {
            "title": "Fresno",
            "paragraphs": [
                {
                    "context": "Fresno is a city.",
                    "qas": [
                        {
                            "id": "q4",
                            "question": "What is Fresno?",
                            "is_impossible": False,
                            "answers": [{"text": "a city", "answer_start": 10}],
                        }
                    ],
                }
            ],
        },
    ],
}


def test_parse_squad_reads_articles_questions_and_unanswerables():
    data = parse_squad(SQUAD)
    assert [a.title for a in data.articles] == ["Normans", "Fresno"]
    assert len(data.articles[0].paragraphs) == 2
    by_id = {q.qid: q for q in data.questions}
    assert by_id["q2"].is_impossible and by_id["q2"].answers == ()
    assert by_id["q3"].para_idx == 1 and by_id["q3"].title == "Normans"


def test_documents_join_paragraphs_and_record_offsets():
    docs = build_documents(parse_squad(SQUAD).articles)
    d = docs[0]
    assert d.text == "The Normans were in Normandy. They came from Denmark.\n\nRollo was a leader."
    assert d.para_starts == [0, len("The Normans were in Normandy. They came from Denmark.") + 2]
    assert [x.doc_id for x in docs] == [0, 1]


def test_gold_spans_point_at_the_answer_text_inside_the_document():
    data = parse_squad(SQUAD)
    docs = {d.title: d for d in build_documents(data.articles)}
    for q in data.questions:
        if q.is_impossible:
            assert gold_spans(q, docs[q.title]) == []
            continue
        for (s, e), a in zip(gold_spans(q, docs[q.title]), q.answers, strict=True):
            assert docs[q.title].text[s:e] == a.text


def test_split_titles_is_seeded_disjoint_and_complete():
    titles = [f"t{i}" for i in range(20)]
    calib, test = split_titles(titles, seed=5, n_calibration=6)
    assert len(calib) == 6 and len(test) == 14
    assert not set(calib) & set(test) and set(calib) | set(test) == set(titles)
    assert split_titles(titles, seed=5, n_calibration=6) == (calib, test)
    assert split_titles(titles, seed=6, n_calibration=6)[0] != calib


def test_sample_questions_is_seeded_restricted_to_titles_and_capped():
    data = parse_squad(SQUAD)
    only = sample_questions(data.questions, ["Fresno"], n=10, seed=1)
    assert [q.qid for q in only] == ["q4"]
    a = sample_questions(data.questions, ["Normans", "Fresno"], n=2, seed=1)
    b = sample_questions(data.questions, ["Normans", "Fresno"], n=2, seed=1)
    assert [q.qid for q in a] == [q.qid for q in b] and len(a) == 2


def test_sample_questions_answerable_only_filter():
    data = parse_squad(SQUAD)
    got = sample_questions(data.questions, ["Normans"], n=10, seed=0, answerable_only=True)
    assert {q.qid for q in got} == {"q1", "q3"}


def test_download_file_verifies_the_checksum(tmp_path: Path):
    src = tmp_path / "src.json"
    src.write_text(json.dumps(SQUAD), encoding="utf-8")
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    dest = tmp_path / "out" / "dev.json"
    assert download_file(src.as_uri(), dest, digest) == dest
    assert sha256_file(dest) == digest
    # a second call reuses the verified file
    assert download_file("file:///does/not/exist.json", dest, digest) == dest


def test_download_file_rejects_a_checksum_mismatch_and_removes_the_file(tmp_path: Path):
    src = tmp_path / "src.json"
    src.write_text("{}", encoding="utf-8")
    dest = tmp_path / "dev.json"
    with pytest.raises(ValueError, match="checksum"):
        download_file(src.as_uri(), dest, "0" * 64)
    assert not dest.exists()
