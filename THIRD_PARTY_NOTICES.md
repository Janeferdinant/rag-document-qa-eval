# Third-party notices

Original code and documentation in this repository are © Jane Rose Ferdinant, licensed CC BY-NC-ND 4.0 (see `LICENSE`).
The components below are not covered by that licence. Each keeps its own licence. Licences and sources were checked on
2026-10-15 at the URLs given. No dataset and no model weights are committed: both are downloaded at run time.

## Dataset

### SQuAD 2.0 (development set)

- Used for: the question-answering corpus (the paragraphs of the 35 dev articles) and the dev questions, including the
  unanswerable ones.
- Source: <https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json>
- Licence: **CC BY-SA 4.0**, <https://creativecommons.org/licenses/by-sa/4.0/>.
- Licence evidence: the dataset home page <https://rajpurkar.github.io/SQuAD-explorer/> says "Download a copy of the
  dataset (distributed under the CC BY-SA 4.0 license)", and the Hugging Face dataset card
  <https://huggingface.co/datasets/rajpurkar/squad_v2> lists `license: cc-by-sa-4.0` (both read on 2026-10-15). The file
  downloads without a login.
- Integrity: the download is verified against SHA-256 `80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8`
  (4,370,528 bytes), pinned in `src/rag_qa_eval/data.py`.
- Attribution: Pranav Rajpurkar, Robin Jia and Percy Liang. 2018. *Know What You Don't Know: Unanswerable Questions for
  SQuAD.* ACL 2018. <https://aclanthology.org/P18-2124/>. The text comes from Wikipedia articles; the authors of the
  original text are credited in the article histories on <https://en.wikipedia.org>.
- Derived files that ARE committed: `docs/data/sample_corpus.json` (verbatim paragraphs of three dev articles, cut into
  overlapping chunks), `docs/data/bm25_index.json` (a token index of those chunks) and `docs/data/examples.json` (a few
  questions and retrieved passages from the dev set, with outputs of the evaluation). These files, and the short
  question and passage excerpts quoted in `docs/results.md` and the README, are shared under **CC BY-SA 4.0** with the
  attribution above, not under the repository licence. `docs/data/LICENSE-DATA.md` repeats this next to the files.
- Held-out use: according to their model cards the reader models below were trained on the SQuAD 2.0 **training** set. The
  dev set was not used to train or tune any model in this repository (only its seeded calibration articles are used to
  choose settings).

## Pretrained models (downloaded from the Hugging Face Hub at run time, never redistributed here)

| Model | Used for | Licence | Source | Notes |
|---|---|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | dense retrieval (default) | Apache-2.0 | <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2> | Authors: Reimers and Gurevych, sentence-transformers. Its model card lists SQuAD 2.0 (87,599 pairs) among the training data. The count is close to the size of the SQuAD training split, so the dev set is probably not included, but the card does not say. The README states this as a possible advantage for dense retrieval. |
| `BAAI/bge-small-en-v1.5` | dense retrieval (comparison) | MIT | <https://huggingface.co/BAAI/bge-small-en-v1.5> | Beijing Academy of Artificial Intelligence (FlagEmbedding). |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | optional reranker | Apache-2.0 | <https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2> | Trained on MS MARCO Passage Ranking, whose own terms (<https://microsoft.github.io/msmarco/>) restrict the data to non-commercial research use. The model licence is Apache-2.0, but anyone using this repository commercially should check the MS MARCO terms. |
| `deepset/minilm-uncased-squad2` | extractive reader (default) | CC BY 4.0 | <https://huggingface.co/deepset/minilm-uncased-squad2> | Attribution: deepset (Vaishali Pal, Branden Chan, Timo Möller, Malte Pietsch, Tanay Soni), fine-tuned from `microsoft/MiniLM-L12-H384-uncased` on SQuAD 2.0. Used unmodified. |
| `deepset/tinyroberta-squad2` | extractive reader (comparison) | CC BY 4.0 | <https://huggingface.co/deepset/tinyroberta-squad2> | Attribution: deepset. Distilled from `deepset/roberta-base-squad2`. Used unmodified. |

CC BY 4.0 licence text: <https://creativecommons.org/licenses/by/4.0/>. Apache-2.0: <https://www.apache.org/licenses/LICENSE-2.0>.
MIT: <https://opensource.org/license/mit>.

## Python packages (installed with pip, not vendored)

| Package | Licence |
|---|---|
| numpy | BSD-3-Clause |
| torch (CPU build) | BSD-3-Clause |
| transformers, huggingface_hub, tokenizers, safetensors | Apache-2.0 |
| sentence-transformers | Apache-2.0 |
| pytest | MIT |
| ruff | MIT |
| mypy | MIT |
| playwright (browser checks only) | Apache-2.0 |

## Methods implemented from the literature (no third-party code copied)

- Okapi BM25 (Robertson and Zaragoza, 2009) with the Lucene-style idf.
- Reciprocal rank fusion (Cormack, Clarke and Buettcher, SIGIR 2009).
- SQuAD 2.0 answer normalisation, exact match and token F1, re-implemented from the published description of the official
  evaluation script.

## External service

- **Google Gemini API** (optional, only with the user's own key, never used in tests or in the committed evaluation).
  Terms: <https://ai.google.dev/gemini-api/terms>. Calls are billed to the user's own Google account, and the terms may
  require a paid service in some regions. Model ids were read from <https://ai.google.dev/gemini-api/docs/models> on
  2026-10-15. This repository contains no Gemini output.
