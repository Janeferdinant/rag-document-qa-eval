# LLM answers with citations: design

This note explains where the LLM answer step plugs into the pipeline and how it relates to the code that was already here. It is the T0 note of the `feat/llm-answers` change.

## Where it plugs in

- `QAPipeline.ask` (in `pipeline.py`) retrieves the top-k chunks with the repo's `Retriever`, then either reads them with the extractive reader (margin and abstention threshold) or with an `LLMClient` through `LLMReader`.
- The evaluation never used the LLM path: `eval_reader.py` measures the extractive reader only, and the committed runs in `docs/runs/` have no LLM numbers.
- This change adds an evaluation stage for the LLM path and a structured answer format. The extractive reader and its numbers are untouched, and they are the baseline.

## What stays as it is

- `llm.py` keeps `LLMClient` (a protocol with `generate(prompt) -> str`), `GeminiClient`, `MockLLMClient`, `LLMReader`, `build_prompt` and `parse_llm_answer`, with their behaviour. `rag-qa ask --llm gemini|mock` and the browser demo keep working. A test pins this.
- The data (the 800 held-out SQuAD 2.0 questions from 25 articles), the retriever (sentence chunks of 160 words, `all-MiniLM-L6-v2`, hybrid + cross-encoder rerank) and the metrics (`qa_metrics.py`) are reused as they are.

## What is added

| Module | Purpose |
|---|---|
| `llm_answers.py` | `CitedAnswer` (`answer`, `citations`, `abstain`), the numbered-passage prompt, validation, and `generate_answer` with one retry. |
| `llm.py` (addition) | `HarnessClient`: an adapter that gives any `evalharness` client the old `generate(prompt)` interface, so the old reader and CLI can use Claude Code, Codex, Ollama and the rest. |
| `llm_metrics.py` | Citation precision and recall (the cited passage contains a gold answer span). Abstention statistics reuse `qa_metrics.abstention_stats`. |
| `eval_llm.py` | The `llm` stage: runs each arm on the same questions, scores it, and writes `docs/runs/<date>-llm.json` and `.md`. |
| `tracing.py` | Optional OpenTelemetry tracing (`pip install ".[tracing]"`, `RAG_TRACING=1`). |

## Dependencies and licences

The model clients, response cache and bootstrap statistics come from `llm-eval-harness` (the optional `llm` extra, installed from the `v0.1.0` tag). It is the author's own library under CC BY-NC-ND 4.0 and is used as an installed dependency only: no code is copied here, so this repository stays CC BY-SA 4.0. `THIRD_PARTY_NOTICES.md` lists it.

## Python version

The floor moves from 3.10 to 3.11, because the harness needs 3.11 and Python 3.10 reaches end of life in October 2026. Baseline before any change (Python 3.11): 134 tests passed and 1 skipped, `ruff check`, `ruff format --check` and `mypy` clean.

## Arms

The extractive reader (baseline); `qwen3:4b` and `qwen3:8b` on all 800 questions; Claude Code and Codex CLI on a fixed 100-question sample, stratified by answerable or unanswerable with seed 0. The local models are also scored on the same 100 for paired comparisons. Hosted arms run through the vendors' own command line tools, so they measure the model plus each tool's own system prompt.
