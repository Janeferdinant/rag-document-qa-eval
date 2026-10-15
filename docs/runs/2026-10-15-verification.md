# Verification, 2026-10-15

Every command in the README's Getting started section was run in a fresh clone of the pushed repository
(`git clone` from GitHub into an empty folder) with a new virtual environment (Python 3.10.6, Windows 11, 16 logical
cores, CPU only, no GPU).

`make` is not installed on this machine, so each Makefile recipe line was run directly. The recipes are one-line
wrappers (for example `make test` is `python -m pytest -q`), so the commands below are what `make` would run.

| README step | Command run | Result |
|---|---|---|
| `make install-dev` | `pip install -r requirements.txt -r requirements-dev.txt -e .` | installed |
| `make test` | `python -m pytest -q` | 134 passed, 1 skipped on the first run; re-running with `-rs` in the same clone gave 134 passed and no skip. I did not find out which test was skipped the first time (the only tests that can skip are the Node parity checks and the browser checks). |
| `make lint` | `ruff check .` and `ruff format --check .` | All checks passed; 43 files already formatted |
| `make typecheck` | `mypy` | Success: no issues found in 20 source files |
| `make data` | `rag-qa ingest` | SQuAD 2.0 dev downloaded, SHA-256 verified; 35 articles, 1,204 paragraphs, 11,873 questions (5,945 unanswerable) |
| `make smoke` | `rag-qa evaluate --smoke --stage all` | retrieval 87 s, reader 39 s, latency 10 s; logs written to `.tmp/smoke-runs` |
| `make e2e` | `pytest -q -m e2e -o addopts="" tests/e2e` (Playwright 1.64.0, Chromium) | 12 passed (6 checks at 390 px and at 1280 px) |
| `rag-qa ask "Where does the Rhine begin?"` | extractive reader, no key | answer "Swiss canton of Graubünden", cited chunk [1] from the Rhine article |
| `rag-qa ask ... --llm gemini` without a key (development checkout, not the fresh clone) | | exits with a message asking for `GEMINI_API_KEY`, no request made |

Deviations from "run every command verbatim":

- The long stages (`make evaluate`) were not re-run in the fresh clone. The smoke setting was used there instead
  (tiny settings: 4 calibration articles, 40 and 60 questions, one chunk size, one embedder). The committed full runs were
  produced earlier the same day in the development checkout with the commands below.
- `rag-qa report` was re-run in the fresh clone on the committed run logs and reproduced the README blocks and
  `docs/results.md` (only line endings differed on Windows).

## The committed full runs

| Stage | Command | Wall time | Code commit recorded in the log |
|---|---|---|---|
| retrieval | `rag-qa evaluate --stage retrieval` | 1,386 s | e914c24 |
| reader | `rag-qa evaluate --stage reader --readers deepset/minilm-uncased-squad2 deepset/tinyroberta-squad2` | 1,479 s | e914c24 |
| latency | `rag-qa evaluate --stage latency` | 89 s | e914c24 |

Each stage is a separate run under the 25-minute limit; the retrieval and reader stages used 23.1 and 24.7 minutes.
The retrieval stage was started at commit 245000c; the only source change between that commit and e914c24 is the default
Gemini model id constant, which the evaluation does not use (the log records the commit at the time the log was
written). Seed 13 for the article split and question samples; 2,000 bootstrap resamples for every interval.

No call was made to the Gemini API or to any other paid or keyed service. The only network use was downloading the
SQuAD 2.0 dev file and the five Hugging Face models (anonymous downloads), and the Playwright tests route every
request to Google's host to a mock and abort any other external request.

## CI

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs ruff, mypy and the fast pytest suite only. It passed on
`main` for the commit "feat: committed evaluation runs, report, static demo and browser checks" (1 min 30 s).
