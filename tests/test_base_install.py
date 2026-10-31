"""The base install (no `llm` or `tracing` extra) must import, report and run its fast checks."""

from __future__ import annotations

import subprocess
import sys

BLOCK = (
    "import sys\n"
    "for name in ('evalharness', 'pydantic', 'opentelemetry'):\n"
    "    sys.modules[name] = None  # makes `import name` raise ImportError\n"
)


def run_blocked(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-c", BLOCK + code], capture_output=True, text=True, check=False)


def test_report_and_llm_report_import_without_the_optional_packages() -> None:
    res = run_blocked("import rag_qa_eval.report, rag_qa_eval.llm_report, rag_qa_eval.cli")
    assert res.returncode == 0, res.stderr[-500:]


def test_llm_answers_fails_clearly_not_silently_without_the_extra() -> None:
    res = run_blocked("import rag_qa_eval.llm_answers")
    assert res.returncode != 0 and "evalharness" in res.stderr
