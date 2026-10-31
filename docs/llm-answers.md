# LLM answers with citations: what was measured and what went wrong on the way

This note belongs to the `llm` evaluation stage (`rag-qa evaluate --stage llm`). The design is in [`llm-answers-design.md`](llm-answers-design.md); the numbers are in the run log [`runs/2026-10-31-llm.md`](runs/2026-10-31-llm.md) and in the README table. Every figure below comes from the run logs in `docs/runs/` or from the probe output quoted in the section on the prompt. Status labels: **Verified** means a run log backs it; **Not tested** means it was not measured.

## What the stage does

For each of the 800 test questions the stage retrieves the top 5 passages with the repo's best retriever (hybrid fusion plus the cross-encoder reranker, sentence chunks of 160 words), asks a model for `{answer, citations, abstain}` with a JSON schema, and checks the reply: citations must be passage numbers from 1 to 5, an abstention has no answer, and an answer needs at least one citation. An invalid reply gets one retry with the error message appended; if it is still invalid the question counts as wrong and is reported in the `Invalid` column, never dropped. The extractive reader's results on the same questions are the baseline, and differences are paired bootstrap intervals on the same questions.

Citation precision is the share of cited passages that contain a gold answer span; citation recall is the share of passages containing a gold span that the answer cites. Both are computed over answerable questions only, because an unanswerable question has no gold span to cite (counting an answered unanswerable question as precision 0 would double-count the missed abstention). They are undefined, and left out of the mean, when there is nothing to cite or nothing to find. The extractive baseline reads one passage, not five, so its precision is whether that one passage holds a gold span. Faithfulness (0 to 2) is judged by `qwen3:8b` over the cited passages on answered questions. That judge was calibrated in another project for answer correctness (weighted kappa 0.835, 95% CI 0.682 to 0.945, on 100 labelled answers), **not** for this rubric, so the faithfulness column is a rough check. The `qwen3:8b` arm is also judged by `qwen3:8b`, which may flatter it. Both points are **Not tested**.

## Two things that were fixed after looking at results

**The first prompt made `qwen3:4b` abstain on everything.** The first full run used a system prompt that stressed abstention ("Never answer from memory"). `qwen3:4b` abstained on 99.2% of the unanswerable questions but also on 377 answerable ones whose answer was in the retrieved passages (abstention recall 0.992, and 4 correct answers out of 800). That run's log is kept as [`runs/2026-10-30-llm-prompt-v1.md`](runs/2026-10-30-llm-prompt-v1.md). Before changing anything I tried three wordings on the 60 questions of a seeded **calibration** sample (never the test questions), using `scripts/prompt_probe.py`. Of those 60, 28 are answerable with the answer in the top 5 and 32 are unanswerable. Cells are "abstained / asked":

| Model | Prompt wording | Answerable, answer in top 5 | Unanswerable |
|---|---|---|---|
| qwen3:4b | original | 28 / 28 | 32 / 32 |
| qwen3:4b | answer-first (adopted) | 8 / 28 | 25 / 32 |
| qwen3:4b | check-then-answer | 15 / 28 | 28 / 32 |
| qwen3:8b | original | 0 / 28 | 5 / 32 |
| qwen3:8b | answer-first (adopted) | 0 / 28 | 4 / 32 |
| qwen3:8b | check-then-answer | 0 / 28 | 11 / 32 |

Status: **Verified** (one run of the probe on 2026-10-30; 60 questions is a small sample and the three wordings are not an exhaustive search). The adopted wording asks the model to abstain "only when none of the passages contains the answer". It is pinned by a test, and its hash is in the run log.

**A parser that was too strict.** With the new prompt, 56.5% of `qwen3:4b`'s replies were invalid ([`runs/2026-10-30-llm-strict-parser.md`](runs/2026-10-30-llm-strict-parser.md)). Every one had the same shape: `{"answer": "abstain", "citations": [], "abstain": true}`. The model had set the flag and written the word into the answer field. The parser now reads `abstain: true` with a placeholder word (`abstain`, `none`, `n/a`, `no answer`, `unanswerable`) as an abstention. A real answer text with `abstain: true` is still invalid. The final run has 0 invalid replies for both arms. This rule was written after seeing results, so the invalid rate of the first run and the final one are both kept in the repository.

## What the final local run shows

On all 800 test questions, paired with the baseline, both models are below the extractive reader: F1 difference -0.092 [-0.129, -0.053] for `qwen3:4b` and -0.312 [-0.357, -0.271] for `qwen3:8b`. On the 100-question sample the same comparison for `qwen3:4b` is -0.031 [-0.127, 0.061], which cannot be told apart from zero: the sample is too small to show a gap of this size, and the 800-question result is the one to quote. The gap for `qwen3:8b` is large in both.

The reason is abstention more than reading. On the 394 unanswerable test questions `qwen3:8b` abstained on 23 and answered 371, while `qwen3:4b` abstained on 303 and answered 91 (the extractive reader abstained on 314). In the other direction `qwen3:4b` abstained on 140 answerable questions whose answer was in the retrieved passages (over-abstention); `qwen3:8b` abstained on none. When the models do answer, citations are mostly sound: citation precision over answerable questions is 0.895 [0.862, 0.928] for `qwen3:4b` and 0.870 [0.841, 0.899] for `qwen3:8b`, against 0.907 [0.875, 0.939] for the extractive reader's single passage.

Both models ran in Ollama on the machine's GPU (the run log records "100% GPU" for each, with `num_ctx` 8192), so the latencies (p50 0.46 s and 0.56 s) are not comparable with the CPU latencies of the other stages. They are the times recorded for the calls that filled the response cache, on a machine that was in use, so treat them as indicative.

Whether hosted models do better is **Not tested** here until their runs are added.

## Five real failures

Each is in the committed run log (`runs/2026-10-31-llm.md`, error analysis) under its id. Questions and passages are SQuAD 2.0 dev items (Wikipedia text, CC BY-SA 4.0, see `THIRD_PARTY_NOTICES.md`).

1. **Over-abstention (`qwen3:4b`, `3c334047`).** "When did the four rightly guided Caliphs die?" Gold: the 7th century. The evaluation found the answer in the retrieved passages; the model abstained. 140 of the 800 test questions end this way for this model.
2. **Right place, wrong claim (`qwen3:4b`, `5e00882d`).** "In which French city was about 2,000 Huguenots killed?" is unanswerable in SQuAD 2.0. The model answered "Paris" and cited the passage about the St. Bartholomew's Day Massacre, which says Catholics "killed thousands of Huguenots in Paris". The number in the question does not match, and the model did not notice.
3. **Answering an unanswerable question (`qwen3:8b`, `4e3e6c42`).** "Who constructed an experiment where stones and rocks were rolled down an incline?" is unanswerable. The model answered "Galileo Galilei" and cited passage 1, whose excerpt in the log begins with Aristotle's explanation of projectile motion. `qwen3:8b` does this on 371 of the 394 unanswerable test questions.
4. **Right answer, citation without it (`qwen3:4b`, `235fa899`).** "What is the estimate for the amount of tree species in the amazon tropical rain forest?" Gold: 16,000. The model answered "16,000" and cited passage 1, in which the evaluation found no gold answer span (its excerpt is about bird, fish and invertebrate species). Exact match scores this as correct; the citation check does not (3 such answers for `qwen3:4b`).
5. **A defensible phrase that exact match rejects (`qwen3:4b`, `26e84521`).** "What is the repulsive force of close range atom interaction?" Gold: "normal force". The model answered "Pauli repulsion" from the cited passage, which names both. Strict string matching counts this as wrong, one reason F1 and exact match understate answers like this.

## Limits

- Two local models, one prompt wording chosen on 60 questions, one article split and seed (13). The comparison says nothing about other prompts, other sizes or other model families. The final run reused cached completions from the earlier full runs with the same prompt and parser, so its latencies are from those earlier calls.
- Exact match and F1 are SQuAD's string measures: `3` against gold `three` counts as wrong, and a correct but longer answer sentence loses F1.
- Faithfulness uses an uncalibrated judge (see above).
- Citation precision is computed against gold answer spans, not against entailment: a passage can contain the answer string without supporting the claim.
- The browser demo shows `qwen3:4b` answers for six precomputed examples. It never calls a model when a visitor clicks. If a client raises a `ValueError`, that call's tokens and cost are not counted; whether any such call happened in the committed runs was not checked (they have no invalid replies).
