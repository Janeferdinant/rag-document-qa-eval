"""Question answering over a corpus: retrieve, read, abstain, cite."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .chunking import Chunk
from .data import Document
from .evalcommon import chunk_corpus
from .llm import LLMClient, LLMReader
from .reader import ReaderOutput, decide
from .retrieval import Retriever


@dataclass(frozen=True)
class Citation:
    rank: int  # 1-based position in the retrieved list
    doc_title: str
    text: str
    start: int
    end: int


@dataclass
class AskResult:
    question: str
    answer: str | None  # None means the system abstained
    cited: Citation | None
    retrieved: list[Citation]
    margin: float | None = None
    mode: str = "extractive"
    notes: list[str] = field(default_factory=list)


def documents_from_files(paths: list[Path]) -> list[Document]:
    docs = []
    for i, p in enumerate(paths):
        text = p.read_text(encoding="utf8", errors="replace").strip()
        docs.append(Document(i, p.name, text, [0]))
    return docs


class QAPipeline:
    def __init__(
        self,
        docs: list[Document],
        retriever: Retriever,
        chunks: list[Chunk],
        reader: Any | None = None,
        llm: LLMClient | None = None,
        method: str = "hybrid",
        k: int = 3,
        threshold: float = 0.0,
    ) -> None:
        if reader is None and llm is None:
            raise ValueError("provide an extractive reader or an LLM client")
        self.docs, self.retriever, self.chunks = docs, retriever, chunks
        self.reader, self.llm = reader, llm
        self.method, self.k, self.threshold = method, k, threshold
        self._titles = {d.doc_id: d.title for d in docs}

    @classmethod
    def build(
        cls,
        docs: list[Document],
        strategy: str,
        size: int,
        encoder: Any,
        reranker: Any | None = None,
        **kwargs: Any,
    ) -> QAPipeline:
        chunks = chunk_corpus(docs, strategy, size)
        retriever = Retriever(chunks, encoder=encoder, reranker=reranker)
        return cls(docs, retriever, chunks, **kwargs)

    def _cite(self, rank: int, pos: int) -> Citation:
        c = self.chunks[pos]
        return Citation(rank, self._titles.get(c.doc_id, ""), c.text, c.start, c.end)

    def ask(self, question: str) -> AskResult:
        positions = self.retriever.search(question, self.method, self.k)
        retrieved = [self._cite(i + 1, p) for i, p in enumerate(positions)]
        if self.llm is not None:
            ans = LLMReader(self.llm).answer(question, [c.text for c in retrieved])
            cited = retrieved[ans.chunk_index] if ans.chunk_index is not None else None
            return AskResult(question, ans.text, cited, retrieved, None, mode=self.llm.name)
        assert self.reader is not None
        outs: list[ReaderOutput] = self.reader.read_many([(question, c.text) for c in retrieved])
        ans2 = decide(outs, self.threshold)
        cited = retrieved[ans2.chunk_index] if ans2.chunk_index is not None else None
        margin = ans2.margin if ans2.margin != float("-inf") else None
        return AskResult(question, ans2.text, cited, retrieved, margin)
