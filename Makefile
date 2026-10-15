PY ?= python

.PHONY: install install-dev data retrieval reader latency evaluate report demo-data smoke test lint typecheck e2e serve all

install:
	$(PY) -m pip install -r requirements.txt -e .

install-dev:
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt -e .

# Download SQuAD 2.0 dev (checksum verified) and print the article split
data:
	$(PY) -m rag_qa_eval.cli ingest

# Each stage is a separate run of under 25 minutes on a 16-thread CPU
retrieval:
	$(PY) -m rag_qa_eval.cli evaluate --stage retrieval

reader:
	$(PY) -m rag_qa_eval.cli evaluate --stage reader --readers deepset/minilm-uncased-squad2 deepset/tinyroberta-squad2

latency:
	$(PY) -m rag_qa_eval.cli evaluate --stage latency

evaluate: retrieval reader latency

# Render README blocks and docs/results.md from docs/runs/*.json
report:
	$(PY) -m rag_qa_eval.cli report

# Regenerate the browser demo's sample corpus, BM25 index and precomputed examples
demo-data:
	$(PY) -m rag_qa_eval.cli export-demo

# Tiny settings that exercise every stage in a few minutes; logs go to .tmp/smoke-runs
smoke:
	$(PY) -m rag_qa_eval.cli evaluate --smoke --stage all

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

typecheck:
	$(PY) -m mypy

# Browser checks at 390 px and 1280 px (needs Playwright and Chromium); the Gemini request is mocked
e2e:
	$(PY) -m pytest -q -m e2e -o addopts="" tests/e2e

# Open http://localhost:8000/ to try the demo
serve:
	$(PY) -m http.server 8000 --directory docs

all: lint typecheck test
