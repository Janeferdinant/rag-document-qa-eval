from __future__ import annotations

import pytest

from rag_qa_eval.chunking import (
    Chunk,
    chunk_document,
    chunk_fixed,
    chunk_sentences,
    chunk_sliding,
    split_sentences,
)

TEXT = " ".join(f"w{i}" for i in range(25))  # 25 words


def words(chunk: Chunk) -> list[str]:
    return chunk.text.split()


def test_fixed_chunks_partition_the_text_without_overlap():
    chunks = chunk_fixed(TEXT, size=10)
    assert [len(words(c)) for c in chunks] == [10, 10, 5]
    assert [w for c in chunks for w in words(c)] == TEXT.split()


def test_chunk_offsets_slice_back_to_the_text():
    for c in chunk_fixed(TEXT, size=7) + chunk_sliding(TEXT, size=8, stride=4):
        assert TEXT[c.start : c.end] == c.text


def test_sliding_window_overlaps_and_covers_everything():
    chunks = chunk_sliding(TEXT, size=10, stride=5)
    assert words(chunks[0])[5:] == words(chunks[1])[:5]
    assert chunks[-1].end == len(TEXT)  # tail is covered
    covered = {w for c in chunks for w in words(c)}
    assert covered == set(TEXT.split())


def test_sliding_window_has_no_redundant_tail_window():
    chunks = chunk_sliding(TEXT, size=10, stride=5)  # windows start at 0,5,10 and 10 reaches the end
    assert [c.start for c in chunks] == sorted({c.start for c in chunks})
    assert len(chunks) == 4  # starts 0, 5, 10, 15 (15..25 is the last full window)


def test_sliding_rejects_bad_parameters():
    with pytest.raises(ValueError):
        chunk_sliding(TEXT, size=10, stride=0)
    with pytest.raises(ValueError):
        chunk_sliding(TEXT, size=10, stride=11)
    with pytest.raises(ValueError):
        chunk_fixed(TEXT, size=0)


def test_empty_and_whitespace_text_yield_no_chunks():
    assert chunk_fixed("", size=5) == []
    assert chunk_sliding("   \n ", size=5, stride=2) == []
    assert chunk_sentences("", max_words=5) == []


def test_text_shorter_than_one_window_is_a_single_chunk():
    chunks = chunk_fixed("just three words", size=50)
    assert len(chunks) == 1 and chunks[0].text == "just three words"


def test_sentence_splitter_handles_abbreviations_and_initials():
    text = "Dr. Smith met J. K. Rowling in the U.S. last year. They talked. Was it fun? Yes!"
    sents = [text[a:b] for a, b in split_sentences(text)]
    assert sents == [
        "Dr. Smith met J. K. Rowling in the U.S. last year.",
        "They talked.",
        "Was it fun?",
        "Yes!",
    ]


def test_sentence_splitter_treats_blank_lines_as_boundaries():
    text = "A heading without a period\n\nThen a sentence."
    sents = [text[a:b] for a, b in split_sentences(text)]
    assert sents == ["A heading without a period", "Then a sentence."]


def test_sentence_chunks_never_cut_a_sentence_when_it_fits():
    text = "One two three. Four five six. Seven eight nine. Ten eleven twelve."
    chunks = chunk_sentences(text, max_words=6)
    assert [c.text for c in chunks] == [
        "One two three. Four five six.",
        "Seven eight nine. Ten eleven twelve.",
    ]


def test_sentence_chunks_hard_split_a_sentence_longer_than_the_limit():
    text = " ".join(f"x{i}" for i in range(12)) + "."
    chunks = chunk_sentences(text, max_words=5)
    assert [len(words(c)) for c in chunks] == [5, 5, 2]


def test_chunk_document_assigns_ids_and_doc_ids():
    chunks = chunk_document(7, TEXT, "fixed", size=10, first_chunk_id=100)
    assert [c.chunk_id for c in chunks] == [100, 101, 102]
    assert {c.doc_id for c in chunks} == {7}


def test_chunk_document_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        chunk_document(0, TEXT, "semantic", size=10)


def test_chunk_document_sliding_default_overlap_is_half_the_window():
    a = chunk_document(0, TEXT, "sliding", size=10)
    b = chunk_sliding(TEXT, size=10, stride=5)
    assert [(c.start, c.end) for c in a] == [(c.start, c.end) for c in b]
