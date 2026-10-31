> **Superseded run log, kept for the record:** second full run, adopted prompt but a parser that rejected `abstain` written into the answer field; replaced by the final run.

# LLM answers with citations

- Date: 2026-10-30, seed 13, k 5, prompt hash `533d7560473ccf0e264c73aaf2aeff0064426dcfdc8bd95e48fcfc6267f520b0`
- Hardware: {'os': 'Windows 10', 'cpu': 'AMD64 Family 25 Model 33 Stepping 0, AuthenticAMD', 'logical_cores': 16, 'ram_gb': 34.3, 'gpu': 'none (CPU only)', 'python': '3.11.14', 'torch': '2.14.1+cpu', 'torch_threads': 8, 'transformers': '5.20.0', 'numpy': '2.2.6', 'code_commit': 'ecd6693', 'sentence_transformers': '6.1.0'}
- Wall time: 1517.5 s; test questions 800, sample 100

## Arms

| Arm | Access | Questions | Exact match | F1 | Abstention precision / recall | Citation precision | Citation recall | Faithfulness (0 to 2) | Invalid | p50 latency | Cost per 1,000 (API-equivalent) | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| extractive-baseline | local, CPU | sample (n=100) | 0.710 [0.620, 0.800] | 0.730 [0.643, 0.813] | 0.760 / 0.775 | 0.720 [0.600, 0.840] | n/a | n/a | 0.000 | - | $0.00 | Verified (run 2026-10-30, seed 13) |
| extractive-baseline | local, CPU | all test questions (n=800) | 0.704 [0.672, 0.734] | 0.732 [0.702, 0.761] | 0.771 / 0.797 | 0.723 [0.677, 0.768] | n/a | n/a | 0.000 | - | $0.00 | Verified (run 2026-10-30, seed 13) |
| qwen3:4b | local (Ollama) | sample (n=100) | 0.270 [0.190, 0.350] | 0.279 [0.193, 0.363] | 0.000 / 0.000 | 0.710 [0.564, 0.839] | 0.938 [0.844, 1.000] | 1.897 [1.744, 2.000] | 0.610 | 0.68 s | $0.00 | Verified (run 2026-10-30, seed 13) |
| qwen3:4b | local (Ollama) | all test questions (n=800) | 0.231 [0.201, 0.261] | 0.265 [0.234, 0.294] | 1.000 / 0.008 | 0.656 [0.606, 0.704] | 0.968 [0.946, 0.988] | 1.901 [1.861, 1.939] | 0.565 | 0.66 s | $0.00 | Verified (run 2026-10-30, seed 13) |
| qwen3:8b | local (Ollama) | sample (n=100) | 0.390 [0.300, 0.490] | 0.452 [0.362, 0.548] | 1.000 / 0.061 | 0.452 [0.354, 0.545] | 0.980 [0.939, 1.000] | 1.680 [1.567, 1.794] | 0.000 | 0.56 s | $0.00 | Verified (run 2026-10-30, seed 13) |
| qwen3:8b | local (Ollama) | all test questions (n=800) | 0.347 [0.314, 0.381] | 0.420 [0.388, 0.452] | 0.852 / 0.058 | 0.453 [0.419, 0.486] | 0.964 [0.946, 0.980] | 1.748 [1.710, 1.786] | 0.000 | 0.56 s | $0.00 | Verified (run 2026-10-30, seed 13) |

**Paired difference from the extractive baseline** on the same sample questions (mean [95% CI] of arm minus baseline, paired bootstrap):

| Arm | F1 difference | Exact match difference |
|---|---|---|
| qwen3:4b | -0.451 [-0.566, -0.343] | -0.440 [-0.560, -0.330] |
| qwen3:8b | -0.278 [-0.402, -0.157] | -0.320 [-0.440, -0.200] |

Passages: the top 5 from hybrid+rerank over sentence chunks of 160 words (all-MiniLM-L6-v2). Prompt hash `533d7560473c`. The sample is 100 questions, stratified by answerable or unanswerable, seed 0. Hosted arms run through the vendors' command line tools, so they measure the model plus each tool's own system prompt, and their latency includes the tool's start-up. Cost is the API-equivalent price of the tokens used, not money paid. Invalid answers (no valid cited answer after one retry) count as wrong. Faithfulness is judged by qwen3:8b on answered questions (0 unsupported, 1 partly, 2 fully supported by the cited passages). This judge was not calibrated for this rubric: its calibration measured answer correctness (100 SQuAD 2.0 answers graded against gold answers, labelled by the author; weighted kappa 0.835, 95% CI 0.682 to 0.945; llm-eval-harness v0.1.0 README). Treat the faithfulness column as a rough check, not a validated measurement.

## Error analysis (qwen3:4b)

| category | count | description |
|---|---|---|
| correct | 209 | answerable, answer matches a gold answer and the cited passage contains it |
| wrong_citation | 3 | answerable, answer matches a gold answer but the cited passages do not contain a gold span |
| retrieval_failure | 1 | answerable, but no retrieved passage contains a gold answer string |
| over_abstention | 0 | answerable, a retrieved passage contains the answer, the model abstained |
| wrong_answer | 40 | answerable, the answer does not match a gold answer |
| correct_abstention | 3 | unanswerable, the model abstained |
| missed_abstention | 92 | unanswerable, the model gave an answer |
| invalid_output | 452 | no valid cited answer after one retry |

- `53d37030` correct: What alumni wrote "The Closing of the American Mind"? | gold ['Allan Bloom', 'Allan Bloom'] | model 'Allan Bloom' | cited [1]
- `5bb66595` correct: What did the increase in skilled workers lead to? | gold ['a decrease in the price of skilled labor', 'decrease in the price of skilled labor'] | model 'a decrease in the price of skilled labor' | cited [1]
- `d6115049` correct: How would one write T(n) = 7n2 + 15n + 40 in big O notation?  | gold ['T(n) = O(n2)', 'T(n) = O(n2)'] | model 'O(n^2)' | cited [1]
- `2ecc1f9b` retrieval_failure: What proclamation officially ended limited Huguenot autonomy? | gold ['Edict of Alès', 'Edict of Alès'] | model 'Edict of Nantes' | cited [1]
- `30c83015` correct_abstention: What percentage of whites are hispanic or latino? | gold [] | model None | cited []
- `0e4b1171` correct_abstention: In what year did the second wave of the plague hit the Netherlands? | gold [] | model None | cited []
- `20bc8e94` correct_abstention: How long has the imagery of the mermaid been used by Sigilium? | gold [] | model None | cited []
