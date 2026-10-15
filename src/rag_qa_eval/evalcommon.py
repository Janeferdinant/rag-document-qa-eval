"""Shared pieces of the evaluation stages: settings, data preparation, components, run logs."""

from __future__ import annotations

import ctypes
import datetime as dt
import json
import os
import platform
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .chunking import STRATEGIES, Chunk, chunk_document
from .data import (
    SQUAD_DEV_SHA256,
    SQUAD_DEV_URL,
    Document,
    Question,
    SquadData,
    build_documents,
    download_file,
    gold_spans,
    load_squad,
    sample_questions,
    split_titles,
)
from .models import DEFAULT_EMBEDDER, DEFAULT_RERANKER, EMBEDDERS
from .reader import DEFAULT_READER
from .retrieval import Encoder, Reranker, relevant_chunk_ids

TINY_READER = "deepset/tinyroberta-squad2"


@dataclass(frozen=True)
class Settings:
    seed: int = 13
    n_calib_articles: int = 10
    n_calib_questions: int = 300
    n_test_questions: int = 800
    strategies: tuple[str, ...] = STRATEGIES
    sizes: tuple[int, ...] = (40, 80, 160)
    embedders: tuple[str, ...] = tuple(EMBEDDERS)
    ks: tuple[int, ...] = (1, 3, 5, 10, 20)
    reader_ks: tuple[int, ...] = (1, 3, 5)
    readers: tuple[str, ...] = (DEFAULT_READER,)
    depth: int = 20
    n_boot: int = 2000
    latency_queries: int = 100
    smoke: bool = False

    @classmethod
    def smoke_settings(cls) -> Settings:
        """Tiny configuration that exercises every code path in about a minute."""
        return cls(
            n_calib_articles=4,
            n_calib_questions=40,
            n_test_questions=60,
            strategies=("sliding", "sentence"),
            sizes=(80,),
            embedders=(DEFAULT_EMBEDDER,),
            reader_ks=(1, 3),
            n_boot=300,
            latency_queries=15,
            smoke=True,
        )


@dataclass
class Components:
    """Factories for the (lazy, cached) pretrained models. Tests pass fakes instead."""

    encoder: Callable[[str], Encoder]
    reranker: Callable[[], Reranker]
    reader: Callable[[str], Any]
    _cache: dict[Any, Any] = field(default_factory=dict)

    def get_encoder(self, name: str) -> Encoder:
        if ("enc", name) not in self._cache:
            self._cache[("enc", name)] = self.encoder(name)
        return self._cache[("enc", name)]

    def get_reranker(self) -> Reranker:
        if ("rr",) not in self._cache:
            self._cache[("rr",)] = self.reranker()
        return self._cache[("rr",)]

    def get_reader(self, name: str) -> Any:
        if ("rd", name) not in self._cache:
            self._cache[("rd", name)] = self.reader(name)
        return self._cache[("rd", name)]


def default_components() -> Components:
    from .models import CrossEncoderReranker, SentenceEncoder
    from .reader import ExtractiveReader

    return Components(
        encoder=lambda name: SentenceEncoder(name),
        reranker=lambda: CrossEncoderReranker(DEFAULT_RERANKER),
        reader=lambda name: ExtractiveReader(name),
    )


@dataclass
class Prepared:
    data: SquadData
    docs: list[Document]
    doc_by_title: dict[str, Document]
    calib_titles: list[str]
    test_titles: list[str]

    def sample(self, split: str, n: int, seed: int, answerable_only: bool) -> list[Question]:
        titles = self.calib_titles if split == "calibration" else self.test_titles
        return sample_questions(self.data.questions, titles, n, seed, answerable_only)


def prepare(settings: Settings, data_dir: Path, squad_path: Path | None = None) -> Prepared:
    path = squad_path or download_file(SQUAD_DEV_URL, data_dir / "raw" / "dev-v2.0.json", SQUAD_DEV_SHA256)
    data = load_squad(path)
    docs = build_documents(data.articles)
    calib, test = split_titles([a.title for a in data.articles], settings.seed, settings.n_calib_articles)
    return Prepared(data, docs, {d.title: d for d in docs}, calib, test)


def chunk_corpus(docs: Sequence[Document], strategy: str, size: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for d in docs:
        chunks.extend(chunk_document(d.doc_id, d.text, strategy, size, first_chunk_id=len(chunks)))
    return chunks


def doc_chunk_index(chunks: Sequence[Chunk]) -> dict[int, list[int]]:
    idx: dict[int, list[int]] = defaultdict(list)
    for pos, c in enumerate(chunks):
        idx[c.doc_id].append(pos)
    return idx


def question_relevant_chunks(
    q: Question, prep: Prepared, chunks: Sequence[Chunk], by_doc: dict[int, list[int]]
) -> set[int]:
    doc = prep.doc_by_title[q.title]
    spans = gold_spans(q, doc)
    local = [chunks[p] for p in by_doc[doc.doc_id]]
    return {by_doc[doc.doc_id][i] for i in relevant_chunk_ids(local, doc.doc_id, spans)}


# ---- hardware and run logs -----------------------------------------------------------------


def _ram_gb() -> float | None:
    try:
        if sys.platform == "win32":

            class MemStatus(ctypes.Structure):
                _fields_ = [  # noqa: RUF012
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            m = MemStatus()
            m.dwLength = ctypes.sizeof(MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return round(m.ullTotalPhys / 1e9, 1)
        with open("/proc/meminfo", encoding="utf8") as f:
            return round(int(f.readline().split()[1]) / 1e6, 1)
    except (OSError, AttributeError, ValueError):
        return None


def git_state() -> str:
    """Short commit hash of the code that produced a run (with '+dirty' if it has local edits)."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "src", "pyproject.toml"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return head + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def hardware_info() -> dict[str, Any]:
    import numpy
    import torch
    import transformers

    info: dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or platform.machine(),
        "logical_cores": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "gpu": "none (CPU only)",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "transformers": transformers.__version__,
        "numpy": numpy.__version__,
        "code_commit": git_state(),
    }
    try:
        import sentence_transformers

        info["sentence_transformers"] = sentence_transformers.__version__
    except ImportError:  # pragma: no cover
        pass
    return info


def today() -> str:
    return dt.date.today().isoformat()


def settings_dict(settings: Settings) -> dict[str, Any]:
    return asdict(settings)


def write_run(out_dir: Path, stage: str, result: dict[str, Any], markdown: str) -> tuple[Path, Path]:
    """Write `<date>-<stage>.json` (machine-readable) and `.md` (human run log) to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = result.get("meta", {}).get("date") or today()
    jp, mp = out_dir / f"{stamp}-{stage}.json", out_dir / f"{stamp}-{stage}.md"
    jp.write_text(json.dumps(result, indent=1, sort_keys=False) + "\n", encoding="utf8")
    mp.write_text(markdown, encoding="utf8")
    return jp, mp


def latest_run(run_dir: Path, stage: str) -> dict[str, Any] | None:
    files = sorted(run_dir.glob(f"*-{stage}.json"))
    return json.loads(files[-1].read_text(encoding="utf8")) if files else None


def fmt_ci(cell: Sequence[float], digits: int = 3) -> str:
    mean, lo, hi = cell
    return f"{mean:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
