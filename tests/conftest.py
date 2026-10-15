from __future__ import annotations

import json
from typing import Any

import pytest

from fakes import components, make_squad, settings
from rag_qa_eval.eval_latency import run_latency
from rag_qa_eval.eval_reader import run_reader
from rag_qa_eval.eval_retrieval import run_retrieval
from rag_qa_eval.evalcommon import prepare


@pytest.fixture(scope="session")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    tmp = tmp_path_factory.mktemp("eval")
    squad = tmp / "dev.json"
    squad.write_text(json.dumps(make_squad()), encoding="utf8")
    s, comps = settings(), components()
    prep = prepare(s, tmp, squad_path=squad)
    msgs: list[str] = []
    retrieval = run_retrieval(prep, s, comps, tmp / "cache", msgs.append)
    reader = run_reader(prep, s, comps, tmp / "cache", retrieval, msgs.append)
    latency = run_latency(prep, s, comps, tmp / "cache", retrieval, reader, msgs.append)
    return {"tmp": tmp, "prep": prep, "retrieval": retrieval, "reader": reader, "latency": latency}
