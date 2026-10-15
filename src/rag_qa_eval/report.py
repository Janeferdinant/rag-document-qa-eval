"""Turn the committed run logs (docs/runs/*.json) into README blocks and docs/results.md.

Every number in the README summary, ablation, reader, error-analysis and latency sections is
rendered from these JSON files, so nothing is typed by hand.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .analysis import CATEGORIES
from .eval_latency import latency_markdown
from .eval_reader import reader_markdown
from .eval_retrieval import retrieval_markdown
from .evalcommon import latest_run

BLOCKS = ("summary", "ablation", "topk", "reader", "errors", "latency")


def ci(cell: list[float], digits: int = 3) -> str:
    return f"{cell[0]:.{digits}f} [{cell[1]:.{digits}f}, {cell[2]:.{digits}f}]"


def short(model: str | None) -> str:
    return "-" if model is None else model.split("/")[-1]


def replace_block(text: str, name: str, content: str) -> str:
    start, end = f"<!-- BLOCK:{name}:START -->", f"<!-- BLOCK:{name}:END -->"
    i, j = text.find(start), text.find(end)
    if i < 0 or j < 0 or j < i:
        raise ValueError(f"markers for block {name!r} not found")
    return text[: i + len(start)] + "\n" + content.strip("\n") + "\n" + text[j:]


def _entry(
    retrieval: dict[str, Any], strategy: str, size: int, emb: str | None, method: str, split: str = "test"
) -> dict[str, Any] | None:
    for e in retrieval["entries"]:
        if (e["strategy"], e["size"], e["embedder"], e["method"], e["split"]) == (
            strategy,
            size,
            emb,
            method,
            split,
        ):
            return e
    return None


def _table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _summary(retrieval: dict[str, Any], reader: dict[str, Any], latency: dict[str, Any]) -> str:
    sel = retrieval["selected"]
    best = _entry(retrieval, sel["strategy"], sel["size"], sel["embedder"], sel["method"])
    assert best is not None
    base = _entry(retrieval, sel["strategy"], sel["size"], None, "bm25")
    head, meta = reader["headline"], retrieval["meta"]
    t, o = head["test"], head["oracle_test"]
    rows: list[list[str]] = []
    cfg = f"{sel['method']}, {sel['strategy']} chunks of {sel['size']} words, {short(sel['embedder'])}"
    for key, label in (("recall@5", "Recall@5"), ("mrr@10", "MRR@10"), ("ndcg@10", "nDCG@10")):
        rows.append(["Best retriever", cfg, label, ci(best["metrics"][key])])
    if base is not None and sel["method"] != "bm25":
        rows.append(["BM25 baseline", "same chunks", "MRR@10", ci(base["metrics"]["mrr@10"])])
    rcfg = f"{head['reader'].split('/')[-1]} + {head['method']}, top-{head['k']}, abstain below margin {head['threshold']}"
    for key, label in (
        ("em", "Exact match"),
        ("f1", "F1"),
        ("abstain_precision", "Abstention precision"),
        ("abstain_recall", "Abstention recall"),
        ("answer_in_context", "Answer in retrieved context"),
    ):
        rows.append(["Best reader pipeline", rcfg, label, ci(t[key])])
    rows.append(["Reader upper bound", "gold paragraph given to the reader", "F1", ci(o["f1"])])
    lat = latency["methods"].get(head["method"])
    if lat:
        e2e = lat["end_to_end_ms"]
        rows.append(["Latency (CPU)", f"{head['method']}, top-{head['k']}, one query at a time",
                     "End-to-end p50 / p95", f"{e2e['p50']:.0f} / {e2e['p95']:.0f} ms"])  # fmt: skip
    note = (
        f"Test split: {meta['n_test_questions']} answerable questions for retrieval and "
        f"{reader['meta']['n_test_questions']} questions (answerable and unanswerable) for answering, from "
        f"{meta['n_test_articles']} Wikipedia articles that are disjoint from the "
        f"{len(meta['calibration_articles'])} calibration articles used to choose settings. "
        "95% percentile-bootstrap intervals over questions."
    )
    return (
        _table(["Component", "Configuration", "Metric", "Test result (mean [95% CI])"], rows) + "\n\n" + note
    )


def _ablation(retrieval: dict[str, Any]) -> str:
    sel = retrieval["selected"]
    emb = sel["best_before_reranker"]["embedder"] or retrieval["meta"]["settings"]["embedders"][0]
    rows: list[list[str]] = []
    for strategy in retrieval["meta"]["settings"]["strategies"]:
        for size in retrieval["meta"]["settings"]["sizes"]:
            key = f"{strategy}-{size}"
            row = [strategy, str(size), str(retrieval["chunk_counts"][key]),
                   f"{retrieval['span_cut_rate'][key]['test'] * 100:.1f}%"]  # fmt: skip
            for emb_, method in ((None, "bm25"), (emb, "dense"), (emb, "hybrid")):
                e = _entry(retrieval, strategy, size, emb_, method)
                row.append(ci(e["metrics"]["mrr@10"], 2) if e else "-")
            rows.append(row)
    out = _table(
        ["Chunking", "Words", "Chunks", "Answer span cut", "BM25 MRR@10", f"Dense MRR@10 ({short(emb)})",
         "Hybrid MRR@10"], rows)  # fmt: skip
    # embedder comparison at the selected chunking
    s, z = sel["strategy"], sel["size"]
    emb_rows = []
    for name in retrieval["meta"]["settings"]["embedders"]:
        d, h = _entry(retrieval, s, z, name, "dense"), _entry(retrieval, s, z, name, "hybrid")
        if d and h:
            emb_rows.append([short(name), ci(d["metrics"]["mrr@10"], 2), ci(h["metrics"]["mrr@10"], 2),
                             ci(d["metrics"]["recall@5"], 2), ci(h["metrics"]["recall@5"], 2)])  # fmt: skip
    if len(emb_rows) > 1:
        out += f"\n\nEmbedding models at {s} chunks of {z} words:\n\n" + _table(
            ["Embedder", "Dense MRR@10", "Hybrid MRR@10", "Dense Recall@5", "Hybrid Recall@5"], emb_rows
        )
    return out


def _topk(retrieval: dict[str, Any]) -> str:
    sel = retrieval["selected"]
    gb = sel["best_before_reranker"]
    s, z = gb["strategy"], gb["size"]
    emb = gb["embedder"] or retrieval["meta"]["settings"]["embedders"][0]
    ks = retrieval["meta"]["settings"]["ks"]
    rows = []
    for method, e_name in (("bm25", None), ("dense", emb), ("hybrid", emb), ("hybrid+rerank", emb)):
        e = _entry(retrieval, s, z, e_name, method)
        if e:
            rows.append([method] + [ci(e["metrics"][f"recall@{k}"], 2) for k in ks])
    return (
        f"{s} chunks of {z} words, test split. Recall@k is the share of questions with at least one "
        "retrieved chunk that fully contains a gold answer span in the top k.\n\n"
        + _table(["Retriever"] + [f"Recall@{k}" for k in ks], rows)
    )


def _reader(reader: dict[str, Any]) -> str:
    head = reader["headline"]
    first = reader["readers"][head["reader"]]
    rows = []
    for g in first["grid"]:
        t = g["test"]
        mark = " (selected)" if (g["method"], g["k"]) == (head["method"], head["k"]) else ""
        rows.append([f"{g['method']}{mark}", str(g["k"]), str(g["threshold"]), ci(t["em"], 2), ci(t["f1"], 2),
                     ci(t["abstain_precision"], 2), ci(t["abstain_recall"], 2),
                     ci(t["answer_in_context"], 2)])  # fmt: skip
    o = first["oracle"]["test"]
    rows.append(["oracle paragraph", "-", str(first["oracle"]["threshold"]), ci(o["em"], 2), ci(o["f1"], 2),
                 ci(o["abstain_precision"], 2), ci(o["abstain_recall"], 2), "-"])  # fmt: skip
    out = f"Reader `{head['reader']}` (test split):\n\n" + _table(
        ["Retriever", "k", "Threshold", "EM", "F1", "Abstention precision", "Abstention recall",
         "Answer in context"], rows)  # fmt: skip
    cmp_rows = []
    for name, rd in reader["readers"].items():
        g = next((x for x in rd["grid"] if (x["method"], x["k"]) == (head["method"], head["k"])), None)
        if g:
            t = g["test"]
            cmp_rows.append([name, str(g["threshold"]), ci(t["em"], 2), ci(t["f1"], 2),
                             ci(t["abstain_precision"], 2), ci(t["abstain_recall"], 2),
                             ci(rd["oracle"]["test"]["f1"], 2)])  # fmt: skip
    if len(cmp_rows) > 1:
        out += f"\n\nReaders at the selected retriever ({head['method']}, top-{head['k']}):\n\n" + _table(
            ["Reader", "Threshold", "EM", "F1", "Abstention precision", "Abstention recall",
             "F1 with gold paragraph"], cmp_rows)  # fmt: skip
    t = head["test"]
    out += (
        "\n\nCitation checks (selected pipeline): the cited chunk contains the answer text in "
        f"{ci(t['supported_by_cited_chunk'], 3)} of answers (true by construction for an extractive reader), "
        f"and for answers with F1 >= 0.5 the cited chunk contains a gold answer span in "
        f"{ci(t['citation_correct_among_good_answers'], 3)}."
    )
    return out


def _errors(reader: dict[str, Any], full: bool = False) -> str:
    ea = reader["error_analysis"]
    rows = [[c, str(ea["counts"][c]), ea["descriptions"][c]] for c in CATEGORIES]
    out = _table(["category", "count", "meaning"], rows)
    out += (
        f"\n\nOut of {ea['n_answerable']} answerable test questions, {ea['answerable_retrieval_failures']} fail "
        f"because the answer is not in the retrieved chunks (retrieval failures) and "
        f"{ea['answerable_reading_failures']} fail although it is (reading failures); "
        f"{ea['reading_failures_that_also_fail_with_oracle_paragraph']} of those reading failures also fail "
        "when the reader is given the gold paragraph."
    )
    pick: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    limit = 3 if full else 1
    for e in ea["examples"]:
        if e["category"] in ("correct", "correct_abstention"):
            continue
        if seen.get(e["category"], 0) < limit:
            seen[e["category"]] = seen.get(e["category"], 0) + 1
            pick.append(e)
    if pick:
        out += (
            "\n\nExamples (SQuAD 2.0 questions, ids hashed; text from Wikipedia via SQuAD, CC BY-SA 4.0):\n"
        )
        for e in pick:
            gold = e["gold"][0] if e["gold"] else "(unanswerable)"
            out += (
                f'\n- `{e["id"]}` {e["category"]}: "{e["question"]}" gold: {gold!r}; '
                f"predicted: {e['prediction']!r}; reader correct with the gold paragraph: "
                f"{e['oracle_paragraph_correct']}."
            )
    return out


def _latency(latency: dict[str, Any]) -> str:
    meta = latency["meta"]
    rows = []
    for m, r in latency["methods"].items():
        a, b, c = r["retrieval_ms"], r["reader_ms"], r["end_to_end_ms"]
        rows.append([m, f"{a['p50']:.1f} / {a['p95']:.1f}", f"{b['p50']:.1f} / {b['p95']:.1f}",
                     f"{c['p50']:.1f} / {c['p95']:.1f}"])  # fmt: skip
    hw = meta["hardware"]
    return (
        f"{meta['n_queries']} questions, top-{meta['k']} chunks read, {meta['n_chunks']} chunks indexed, "
        f"batch size 1, warm models. CPU: {hw['cpu']} ({hw['logical_cores']} logical cores, "
        f"{hw['torch_threads']} torch threads), no GPU. Values are milliseconds, p50 / p95.\n\n"
        + _table(["Retriever", "Retrieval", "Reader", "End to end"], rows)
    )


def render_blocks(
    retrieval: dict[str, Any], reader: dict[str, Any], latency: dict[str, Any]
) -> dict[str, str]:
    return {
        "summary": _summary(retrieval, reader, latency),
        "ablation": _ablation(retrieval),
        "topk": _topk(retrieval),
        "reader": _reader(reader),
        "errors": _errors(reader),
        "latency": _latency(latency),
    }


def write_report(runs_dir: Path, readme: Path, results: Path) -> None:
    retrieval = latest_run(runs_dir, "retrieval")
    reader = latest_run(runs_dir, "reader")
    latency = latest_run(runs_dir, "latency")
    if not (retrieval and reader and latency):
        raise FileNotFoundError("run all three evaluation stages first (retrieval, reader, latency)")
    blocks = render_blocks(retrieval, reader, latency)
    text = readme.read_text(encoding="utf8")
    for name in BLOCKS:
        text = replace_block(text, name, blocks[name])
    readme.write_text(text, encoding="utf8")
    results.parent.mkdir(parents=True, exist_ok=True)
    results.write_text(
        "# Results\n\nGenerated by `rag-qa report` from the run logs in `docs/runs/`. Do not edit by hand.\n\n"
        + retrieval_markdown(retrieval).replace("# ", "## ", 1)
        + "\n"
        + reader_markdown(reader).replace("# ", "## ", 1)
        + "\n"
        + latency_markdown(latency).replace("# ", "## ", 1),
        encoding="utf8",
    )
