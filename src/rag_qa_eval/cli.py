"""Command line interface: `rag-qa ingest | evaluate | ask | report | export-demo`."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from .evalcommon import (
    Settings,
    default_components,
    latest_run,
    prepare,
    write_run,
)

DEFAULT_DATA = Path("data")
DEFAULT_RUNS = Path("docs/runs")
SMOKE_RUNS = Path(".tmp/smoke-runs")


def _log(msg: str) -> None:
    print(msg, flush=True)


def cmd_ingest(args: argparse.Namespace) -> int:
    settings = Settings.smoke_settings() if args.smoke else Settings(seed=args.seed)
    prep = prepare(settings, args.data_dir)
    n_q = len(prep.data.questions)
    n_un = sum(q.is_impossible for q in prep.data.questions)
    words = sum(len(d.text.split()) for d in prep.docs)
    _log("SQuAD 2.0 dev set (CC BY-SA 4.0, Rajpurkar, Jia and Liang 2018), checksum verified.")
    _log(f"articles: {len(prep.docs)}, paragraphs: {sum(len(a.paragraphs) for a in prep.data.articles)}")
    _log(f"corpus words: {words:,}; questions: {n_q} ({n_un} unanswerable)")
    _log(f"calibration articles ({len(prep.calib_titles)}): {', '.join(prep.calib_titles)}")
    _log(f"test articles: {len(prep.test_titles)}")
    out = args.data_dir / "processed" / "split.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"seed": settings.seed, "calibration": prep.calib_titles, "test": prep.test_titles}),
        encoding="utf8",
    )
    _log(f"wrote {out}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .eval_latency import latency_markdown, run_latency
    from .eval_reader import reader_markdown, run_reader
    from .eval_retrieval import retrieval_markdown, run_retrieval

    settings = Settings.smoke_settings() if args.smoke else Settings(seed=args.seed)
    if args.n_test:
        settings = replace(settings, n_test_questions=args.n_test)
    if args.readers:
        settings = replace(settings, readers=tuple(args.readers))
    out_dir = args.out or (SMOKE_RUNS if args.smoke else DEFAULT_RUNS)
    cache_dir = args.data_dir / "cache"
    comps = default_components()
    prep = prepare(settings, args.data_dir)
    stages = ["retrieval", "reader", "latency"] if args.stage == "all" else [args.stage]
    for stage in stages:
        t0 = time.perf_counter()
        _log(f"== stage: {stage}")
        retrieval = latest_run(out_dir, "retrieval")
        if stage == "retrieval":
            result = run_retrieval(prep, settings, comps, cache_dir, _log)
            md = retrieval_markdown(result)
        else:
            if retrieval is None:
                _log("run the retrieval stage first")
                return 2
            if stage == "reader":
                result = run_reader(prep, settings, comps, cache_dir, retrieval, _log)
                md = reader_markdown(result)
            else:
                reader_result = latest_run(out_dir, "reader")
                result = run_latency(prep, settings, comps, cache_dir, retrieval, reader_result, _log)
                md = latency_markdown(result)
        jp, mp = write_run(out_dir, stage, result, md)
        _log(f"wrote {jp} and {mp} ({time.perf_counter() - t0:.0f}s)")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from .data import load_squad
    from .evalcommon import chunk_corpus  # noqa: F401
    from .llm import GeminiClient, LLMClient, MissingKeyError, MockLLMClient
    from .models import DEFAULT_EMBEDDER, DEFAULT_RERANKER, CrossEncoderReranker, SentenceEncoder
    from .pipeline import QAPipeline, documents_from_files

    if args.docs:
        docs = documents_from_files([Path(p) for p in args.docs])
    else:
        raw = args.data_dir / "raw" / "dev-v2.0.json"
        if not raw.exists():
            _log("no corpus: pass --docs FILE... or run `rag-qa ingest` first")
            return 2
        from .data import build_documents

        docs = build_documents(load_squad(raw).articles)

    reader_run = latest_run(args.runs, "reader")
    strategy, size, method, k, threshold = "sliding", 80, "hybrid", 3, 0.0
    if reader_run and not args.docs:
        ch, head = reader_run["meta"]["chunking"], reader_run["headline"]
        strategy, size, method, k, threshold = (
            ch["strategy"], ch["size"], head["method"], head["k"], head["threshold"],
        )  # fmt: skip
    if args.method:
        method = args.method
    if args.k:
        k = args.k
    if args.threshold is not None:
        threshold = args.threshold

    llm: LLMClient | None = None
    reader = None
    if args.llm == "gemini":
        try:
            llm = GeminiClient(model=args.model) if args.model else GeminiClient()
        except MissingKeyError as err:
            _log(f"{err}. Set GEMINI_API_KEY to use Gemini (calls are billed to your own account).")
            return 2
    elif args.llm == "mock":
        llm = MockLLMClient()
    else:
        from .reader import ExtractiveReader

        reader = ExtractiveReader()
    encoder = SentenceEncoder(DEFAULT_EMBEDDER)
    reranker = CrossEncoderReranker(DEFAULT_RERANKER) if method == "hybrid+rerank" else None
    pipe = QAPipeline.build(
        docs, strategy, size, encoder, reranker, reader=reader, llm=llm,
        method=method, k=k, threshold=threshold,
    )  # fmt: skip
    res = pipe.ask(args.question)
    if res.answer is None:
        _log("Answer: (abstained: the retrieved passages do not answer this question)")
    else:
        _log(f"Answer: {res.answer}")
    if res.cited:
        _log(f"Cited: [{res.cited.rank}] {res.cited.doc_title}")
    _log(f"Mode: {res.mode}; retrieval: {method}, top-{k}" + (
        f"; margin {res.margin:.2f} (threshold {threshold})" if res.margin is not None else ""
    ))  # fmt: skip
    for c in res.retrieved:
        _log(f"[{c.rank}] ({c.doc_title}) {' '.join(c.text.split())[:300]}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .report import write_report

    write_report(args.runs, args.readme, args.results)
    _log(f"updated {args.results} and the summary block in {args.readme}")
    return 0


def cmd_export_demo(args: argparse.Namespace) -> int:
    from .demo_export import export_demo

    export_demo(args.data_dir, args.runs, args.out)
    _log(f"wrote demo data to {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rag-qa", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    i = sub.add_parser("ingest", help="download SQuAD 2.0 dev (checksum verified) and print the split")
    i.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    i.add_argument("--seed", type=int, default=13)
    i.add_argument("--smoke", action="store_true")
    i.set_defaults(fn=cmd_ingest)

    e = sub.add_parser("evaluate", help="run the evaluation stages and write docs/runs logs")
    e.add_argument("--stage", choices=["retrieval", "reader", "latency", "all"], default="all")
    e.add_argument("--smoke", action="store_true", help="tiny settings; logs go to .tmp/smoke-runs")
    e.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    e.add_argument("--out", type=Path, default=None)
    e.add_argument("--seed", type=int, default=13)
    e.add_argument("--n-test", type=int, default=0, help="override the number of test questions")
    e.add_argument("--readers", nargs="+", default=None)
    e.set_defaults(fn=cmd_evaluate)

    a = sub.add_parser("ask", help="ask one question over the SQuAD corpus or your own text files")
    a.add_argument("question")
    a.add_argument("--docs", nargs="+", default=None, help="text files to use as the corpus")
    a.add_argument("--llm", choices=["none", "gemini", "mock"], default="none")
    a.add_argument("--model", default=None, help="Gemini model id (default in llm.py)")
    a.add_argument("--method", choices=["bm25", "dense", "hybrid", "hybrid+rerank"], default=None)
    a.add_argument("--k", type=int, default=0)
    a.add_argument("--threshold", type=float, default=None)
    a.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    a.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    a.set_defaults(fn=cmd_ask)

    r = sub.add_parser("report", help="write docs/results.md and the README summary from run logs")
    r.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    r.add_argument("--readme", type=Path, default=Path("README.md"))
    r.add_argument("--results", type=Path, default=Path("docs/results.md"))
    r.set_defaults(fn=cmd_report)

    d = sub.add_parser("export-demo", help="write the browser demo's corpus, index and examples")
    d.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    d.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    d.add_argument("--out", type=Path, default=Path("docs/data"))
    d.set_defaults(fn=cmd_export_demo)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
