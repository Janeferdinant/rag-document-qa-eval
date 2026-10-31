> **Superseded run log, kept for the record:** first full run, original prompt (it made qwen3:4b abstain on nearly everything); replaced by the adopted prompt, see `docs/llm-answers.md`.

# LLM answers with citations

- Date: 2026-10-30, seed 13, k 5, prompt hash `8d9c1310451589ae542b0ec98ed06e13647746bc221c8abd2d4024aa70160f3b`
- Hardware: {'os': 'Windows 10', 'cpu': 'AMD64 Family 25 Model 33 Stepping 0, AuthenticAMD', 'logical_cores': 16, 'ram_gb': 34.3, 'gpu': 'none (CPU only)', 'python': '3.11.14', 'torch': '2.14.1+cpu', 'torch_threads': 8, 'transformers': '5.20.0', 'numpy': '2.2.6', 'code_commit': '666eaa6', 'sentence_transformers': '6.1.0'}
- Wall time: 1574.2 s; test questions 800, sample 100

## Arms

| Arm | Access | Questions | Exact match | F1 | Abstention precision / recall | Citation precision | Citation recall | Faithfulness (0 to 2) | Invalid | p50 latency | Cost per 1,000 (API-equivalent) | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| extractive-baseline | local, CPU | sample (n=100) | 0.710 [0.620, 0.800] | 0.730 [0.643, 0.813] | 0.760 / 0.775 | 0.720 [0.600, 0.840] | n/a | n/a | 0.000 | - | $0.00 | Verified (run 2026-10-30, seed 13) |
| extractive-baseline | local, CPU | all test questions (n=800) | 0.704 [0.672, 0.734] | 0.732 [0.702, 0.761] | 0.771 / 0.797 | 0.723 [0.677, 0.768] | n/a | n/a | 0.000 | - | $0.00 | Verified (run 2026-10-30, seed 13) |
| qwen3:4b | local (Ollama) | sample (n=100) | 0.480 [0.380, 0.580] | 0.486 [0.390, 0.586] | 0.500 / 0.980 | 0.500 [0.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [0.000, 2.000] | 0.020 | 0.37 s | $0.00 | Verified (run 2026-10-30, seed 13) |
| qwen3:4b | local (Ollama) | all test questions (n=800) | 0.492 [0.459, 0.527] | 0.493 [0.460, 0.528] | 0.500 / 0.992 | 0.714 [0.429, 1.000] | 1.000 [1.000, 1.000] | 1.714 [1.143, 2.000] | 0.014 | 0.37 s | $0.00 | Verified (run 2026-10-30, seed 13) |
| qwen3:8b | local (Ollama) | sample (n=100) | 0.380 [0.290, 0.480] | 0.447 [0.356, 0.544] | 1.000 / 0.082 | 0.457 [0.360, 0.558] | 0.980 [0.939, 1.000] | 1.667 [1.542, 1.781] | 0.000 | 0.57 s | $0.00 | Verified (run 2026-10-30, seed 13) |
| qwen3:8b | local (Ollama) | all test questions (n=800) | 0.329 [0.296, 0.361] | 0.413 [0.382, 0.443] | 0.844 / 0.069 | 0.443 [0.409, 0.478] | 0.966 [0.948, 0.981] | 1.772 [1.731, 1.810] | 0.001 | 0.58 s | $0.00 | Verified (run 2026-10-30, seed 13) |

**Paired difference from the extractive baseline** on the same sample questions (mean [95% CI] of arm minus baseline, paired bootstrap):

| Arm | F1 difference | Exact match difference |
|---|---|---|
| qwen3:4b | -0.244 [-0.372, -0.117] | -0.230 [-0.360, -0.110] |
| qwen3:8b | -0.283 [-0.408, -0.163] | -0.330 [-0.460, -0.200] |

Passages: the top 5 from hybrid+rerank over sentence chunks of 160 words (all-MiniLM-L6-v2). Prompt hash `8d9c13104515`. The sample is 100 questions, stratified by answerable or unanswerable, seed 0. Hosted arms run through the vendors' command line tools, so they measure the model plus each tool's own system prompt, and their latency includes the tool's start-up. Cost is the API-equivalent price of the tokens used, not money paid. Invalid answers (no valid cited answer after one retry) count as wrong. Faithfulness is judged by qwen3:8b on answered questions (0 unsupported, 1 partly, 2 fully supported by the cited passages). This judge was not calibrated for this rubric: its calibration measured answer correctness (100 SQuAD 2.0 answers graded against gold answers, labelled by the author; weighted kappa 0.835, 95% CI 0.682 to 0.945; llm-eval-harness v0.1.0 README). Treat the faithfulness column as a rough check, not a validated measurement.

## Error analysis (qwen3:4b)

| category | count | description |
|---|---|---|
| correct | 4 | answerable, answer matches a gold answer and the cited passage contains it |
| wrong_citation | 0 | answerable, answer matches a gold answer but the cited passages do not contain a gold span |
| retrieval_failure | 14 | answerable, but no retrieved passage contains a gold answer string |
| over_abstention | 377 | answerable, a retrieved passage contains the answer, the model abstained |
| wrong_answer | 1 | answerable, the answer does not match a gold answer |
| correct_abstention | 391 | unanswerable, the model abstained |
| missed_abstention | 2 | unanswerable, the model gave an answer |
| invalid_output | 11 | no valid cited answer after one retry |

- `bbbe67ea` correct: Besides drugs, what else do specialty pharmacies provide? | gold ['lab monitoring, adherence counseling, and assist patients with cost-containment strategies needed to obtain their expensive specialty drugs', 'lab monitoring, adherence counseling, and assist patients with cost-containment strategies'] | model 'lab monitoring, adherence counseling, and assist patients with cost-containment strategies' | cited [1]
- `8fbba67d` correct: What did Artur Oppman give to the world? | gold ['legend', 'The best-known legend'] | model 'The best-known legend, by Artur Oppman' | cited [1]
- `32c0f1e2` correct: The Sand Bernardino - Riverside area maintains what kind of district? | gold ['business', 'business districts'] | model 'business districts' | cited [1]
- `b971f019` retrieval_failure: What famous civil rights leader called Harvard home? | gold ['W. E. B. Du Bois', 'W. E. B. Du Bois'] | model None | cited []
- `59e064e3` retrieval_failure: What actually causes rigidity in matter? | gold ['the Pauli exclusion principle', 'the Pauli exclusion principle'] | model None | cited []
- `f6dde170` retrieval_failure: What are incompetent government's commitment to social justice limited to? | gold ['rhetoric', 'rhetoric'] | model None | cited []
- `31e775d7` correct_abstention: What Huguenot French family had the largest brandy distilling business in South Africa? | gold [] | model None | cited []
- `4dfff9c1` correct_abstention: Who is the developer of the McKinsey concept? | gold [] | model None | cited []
- `df7e0e64` correct_abstention:  What was Tugh Temur not known for? | gold [] | model None | cited []
