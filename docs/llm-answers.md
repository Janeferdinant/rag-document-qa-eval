# LLM answers with citations: what was measured and what went wrong on the way

This note belongs to the `llm` evaluation stage (`rag-qa evaluate --stage llm`). The design is in [`llm-answers-design.md`](llm-answers-design.md); the numbers are in the run log [`runs/2026-10-30-llm.md`](runs/2026-10-30-llm.md) and in the README table. Every figure below comes from those logs or from the probe output quoted in the section on the prompt. Status labels: **Verified** means a run log backs it; **Not tested** means it was not measured.

## What the stage does

For each of the 800 test questions the stage retrieves the top 5 passages with the repo's best retriever (hybrid fusion plus the cross-encoder reranker, sentence chunks of 160 words), asks a model for `{answer, citations, abstain}` with a JSON schema, and checks the reply: citations must be passage numbers from 1 to 5, an abstention has no answer, and an answer needs at least one citation. An invalid reply gets one retry with the error message appended; if it is still invalid the question counts as wrong and is reported in the `Invalid` column, never dropped. The extractive reader's results on the same questions are the baseline, and differences are paired bootstrap intervals on the same questions.

Citation precision is the share of cited passages that contain a gold answer span; citation recall is the share of passages containing a gold span that the answer cites. Both are undefined (shown as `n/a` or left out of the mean) when there is nothing to cite or nothing to find. Faithfulness (0 to 2) is judged by `qwen3:8b` over the cited passages on answered questions. That judge was calibrated in another project for answer correctness (weighted kappa 0.835, 95% CI 0.682 to 0.945, on 100 labelled answers), **not** for this rubric, so the faithfulness column is a rough check. The `qwen3:8b` arm is also judged by `qwen3:8b`, which may flatter it. Both points are **Not tested**.

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

On the 100-question sample (paired with the baseline), `qwen3:4b` is not distinguishable from the extractive reader: F1 difference -0.031 [-0.127, 0.061]. `qwen3:8b` is clearly lower: -0.278 [-0.402, -0.157]. The reason is abstention, not reading. On the 394 unanswerable test questions `qwen3:8b` abstained on 23 and answered 371, while `qwen3:4b` abstained on 303 and answered 91. In the other direction `qwen3:4b` abstained on 140 answerable questions whose answer was in the retrieved passages (over-abstention); `qwen3:8b` abstained on none. Latency is Ollama on the same desktop as the other stages, which was in use during the run, so treat it as indicative; whether Ollama used the GPU was not recorded.

Whether hosted models do better is **Not tested** here until their runs are added.

## Five real failures

Questions and passages are SQuAD 2.0 dev items (Wikipedia text, CC BY-SA 4.0, see `THIRD_PARTY_NOTICES.md`). Answers are the models' outputs read back from the response cache.

1. **Over-abstention (`qwen3:4b`).** "When did the Normans attack Dyrrachium?" Gold: 1185. A retrieved passage contains the answer; the model abstained. `qwen3:8b` answered "In 1107, the Normans under the command of Bohemond..." (a different event, cited passage 1), so the larger model is not simply better: it answered with the wrong year.
2. **Answering from memory with an unrelated citation (`qwen3:4b`).** "Who was the first American to travel to Amazon River" is unanswerable in SQuAD 2.0. The model answered "Francisco de Orellana" and cited passage 1, which is about the population density the rainforest can sustain. The answer is not in that passage, and the citation made it look supported.
3. **Swapped word in an unanswerable question (`qwen3:8b`).** "What extended as far north as 45°?" The passage says the rainforest "extended as far south as 45°". The model answered "the tropical rainforest" and cited it. SQuAD 2.0's unanswerable questions are written to look like answerable ones, and this is the failure the abstention columns measure.
4. **Right answer, citation without the answer (`qwen3:4b`).** "How many organizations issued the joint statement on climate change?" Gold: 16. The model answered "16" and cited passage 2, in which the evaluation found no gold answer span (the passage lists the academies). Exact match scores this as correct; the citation check does not.
5. **Copying from the question (`qwen3:4b`).** "Which region of California is Palm Springs located in?" Gold: southern. The model answered "Palm Springs" and cited passage 1, which talks about southern California.

## Limits

- Two local models, one prompt wording chosen on 60 questions, one article split and seed (13). The comparison says nothing about other prompts, other sizes or other model families.
- Exact match and F1 are SQuAD's string measures: `3` against gold `three` counts as wrong, and a correct but longer answer sentence loses F1.
- Faithfulness uses an uncalibrated judge (see above).
- Citation precision is computed against gold answer spans, not against entailment: a passage can contain the answer string without supporting the claim.
- The browser demo shows `qwen3:4b` answers for six precomputed examples. It never calls a model when a visitor clicks.
