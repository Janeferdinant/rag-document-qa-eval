"""How often do local models abstain under three prompt wordings? Runs on the CALIBRATION split only, never the test set.

    python scripts/prompt_probe.py qwen3:4b qwen3:8b

Needs Ollama and the committed retrieval and reader runs. Results are in docs/llm-answers.md.
"""

import json
import sys
from pathlib import Path

from evalharness.llm.base import Message, get_client

from rag_qa_eval import llm_answers as la
from rag_qa_eval.data import gold_spans
from rag_qa_eval.demo_export import settings_from_dict
from rag_qa_eval.eval_llm import llm_retriever
from rag_qa_eval.evalcommon import default_components, latest_run, prepare
from rag_qa_eval.retrieval import chunk_contains_span

root = Path(".")
runs = root / "docs" / "runs"
data_dir = root / "data"
reader = latest_run(runs, "reader")
retrieval = latest_run(runs, "retrieval")
settings = settings_from_dict(reader["meta"]["settings"])
prep = prepare(settings, data_dir)
comps = default_components()
chunks, retriever, method, _, _ = llm_retriever(prep, settings, comps, data_dir / "cache", retrieval)
qs = prep.sample("calibration", 60, 1, False)
ranked = retriever.search_batch([q.text for q in qs], method, 5)

ORIGINAL = (
    "You answer questions using only the numbered passages you are given. "
    "Reply with the shortest span that fully answers the question, copied from a passage, "
    "and cite the number of every passage the answer relies on (the numbers are 1-based, as shown). "
    "If the passages do not contain the answer, abstain: set `abstain` to true, `answer` to an empty "
    "string and `citations` to an empty list. Never answer from memory."
)
VARIANTS = {
    "original": ORIGINAL,
    "answer-first (adopted)": la.SYSTEM_PROMPT,
    "check-then-answer": (
        "You answer questions using only the numbered passages you are given. First look for a passage that "
        "states the answer. If one does, reply with the shortest span from it that fully answers the question "
        "and cite the passage number(s) (1-based, as shown). If no passage states the answer, abstain: set "
        "`abstain` to true, `answer` to an empty string and `citations` to an empty list."
    ),
}
out = {}
for model in sys.argv[1:]:
    client = get_client("ollama")
    for name, system in VARIANTS.items():
        stats = {"ans_supported": [0, 0], "unans": [0, 0]}  # [abstained, total]
        for q, top in zip(qs, ranked, strict=True):
            doc = prep.doc_by_title[q.title]
            spans = gold_spans(q, doc)
            supp = any(chunk_contains_span(chunks[p], doc.doc_id, spans) for p in top)
            if not q.is_impossible and not supp:
                continue
            msgs = la.build_messages(q.text, [chunks[p].text for p in top])
            msgs[0] = Message(role="system", content=system)
            c = client.complete(
                msgs, model=model, temperature=0.0, seed=0, max_tokens=1024, response_schema=la.CitedAnswer
            )
            abstain = bool((c.data or {}).get("abstain", True))
            key = "unans" if q.is_impossible else "ans_supported"
            stats[key][0] += abstain
            stats[key][1] += 1
        out[f"{model} / {name}"] = stats
        print(model, name, stats, flush=True)
print(json.dumps(out))
