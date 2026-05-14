# Reliability-aware Abstention E2 — 20260513

Source: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260429_everpass_debug_bs4/gsm8k_gen128_steps64_vote_confidence_gap_answer_window5_mean_rawsum_bs4_debug/rank_0_generations.json`
Transfer: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260429_everpass_debug_bs4/svamp_gen128_steps64_vote_confidence_gap_answer_window5_mean_rawsum_bs4_debug/rank_0_generations.json`
outer split seed=42; val_frac=0.8; inner_seed=7; inner_frac=0.8

## Models

- `zsum_combined`: Phase E signed z-score sum baseline
- `best_single`: `max_gap`
- `logistic_top3`: NumPy logistic regression on ['max_gap', 'n_unique_answers', 'last_change_step_frac']
- `logistic_broad`: NumPy logistic regression on ['max_gap', 'mean_gap', 'last_change_gap', 'std_gap', 'n_valid', 'n_unique_answers', 'last_change_step_frac', 'max_gap_step_frac', 'first_appear_step_frac']

## Inner-dev L2 selection

| Model | L2 | Dev AURC | Dev SelAcc@80% |
|---|---:|---:|---:|
| logistic_top3 **best** | 0.0 | 0.2252 | 78.11% |
| logistic_top3 | 0.01 | 0.2252 | 78.11% |
| logistic_top3 | 0.1 | 0.2266 | 77.51% |
| logistic_top3 | 1.0 | 0.2253 | 77.51% |
| logistic_top3 | 5.0 | 0.2260 | 76.92% |
| logistic_top3 | 10.0 | 0.2260 | 76.92% |
| logistic_broad | 0.0 | 0.2080 | 79.88% |
| logistic_broad **best** | 0.01 | 0.2077 | 79.88% |
| logistic_broad | 0.1 | 0.2080 | 81.07% |
| logistic_broad | 1.0 | 0.2152 | 79.29% |
| logistic_broad | 5.0 | 0.2136 | 79.29% |
| logistic_broad | 10.0 | 0.2147 | 79.29% |

## GSM8K Validation

| Score | AURC | SelAcc@80% | SelAcc@50% | Cov@90%acc |
|---|---:|---:|---:|---:|
| zsum_combined | 0.2187 | 78.08% | 86.74% | n/a |
| best_single (max_gap) | 0.2309 | 76.54% | 83.71% | n/a |
| logistic_top3 | 0.2218 | 77.49% | 86.55% | n/a |
| logistic_broad | 0.2103 | 79.03% | 88.45% | n/a |
| oracle | 0.1234 | 88.27% | 100.00% | — |
| random | 0.2985 | 69.55% | 70.08% | — |

## GSM8K Test (one-shot)

| Score | AURC | SelAcc@80% | SelAcc@50% | Cov@90%acc |
|---|---:|---:|---:|---:|
| zsum_combined | 0.2515 | 74.88% | 81.06% | n/a |
| best_single (max_gap) | 0.2626 | 72.99% | 79.55% | n/a |
| logistic_top3 | 0.2408 | 75.36% | 83.33% | n/a |
| logistic_broad | 0.2354 | 77.73% | 83.33% | n/a |
| oracle | 0.1521 | 84.36% | 100.00% | — |
| random | 0.3034 | 70.62% | 69.70% | — |

## SVAMP transfer

| Score | AURC | SelAcc@80% | SelAcc@50% | Cov@90%acc |
|---|---:|---:|---:|---:|
| zsum_combined | 0.0954 | 91.67% | 91.33% | 90% |
| best_single (max_gap) | 0.0898 | 90.83% | 95.33% | 80% |
| logistic_top3 | 0.0802 | 92.92% | 95.33% | 90% |
| logistic_broad | 0.0848 | 92.50% | 94.00% | 90% |
| oracle | 0.0298 | 100.00% | 100.00% | — |
| random | 0.1366 | 86.25% | 86.67% | — |

## Notes

- Logistic models are trained only on the GSM8K validation split, with L2 chosen on an inner dev split.
- Test and SVAMP scores see the frozen val-fit model only; no refit is performed outside validation.
- This pass still targets baseline `exp_only` correctness, so improvements here should be read as sample-level reliability gains rather than voting gains.
