# GSM8K Commit-Gap Proxy (Phase 8 offline) — 20260513

Source: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260429_everpass_debug_bs4/gsm8k_gen128_steps64_vote_confidence_gap_answer_window5_mean_rawsum_bs4_debug/rank_0_generations.json`
Transfer: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260429_everpass_debug_bs4/svamp_gen128_steps64_vote_confidence_gap_answer_window5_mean_rawsum_bs4_debug/rank_0_generations.json`
alpha=5.0; seed=42; validation_frac=0.8

## Proxy single-event vote — Validation (n=1055)

| Method | Val Acc |
|---|---:|
| exp_only (baseline) | 70.62% |
| proxy_first_appear | 34.41% |
| proxy_last_change | 69.95% |
| proxy_max_gap | 69.86% |

## Proxy single-event vote — Test (n=264)

| Method | Test Acc |
|---|---:|
| exp_only (baseline) | 67.42% |
| proxy_first_appear | 34.47% |
| proxy_last_change | 67.05% |
| proxy_max_gap | 67.80% |

## Proxy single-event vote — Transfer SVAMP (n=300)

| Method | SVAMP Acc |
|---|---:|
| exp_only (baseline) | 86.33% |
| proxy_first_appear | 72.33% |
| proxy_last_change | 86.00% |
| proxy_max_gap | 86.00% |

## Per-sample feature correlation with baseline correctness

n=1319, correct_rate=69.98%

| Feature | Pearson | Spearman |
|---|---:|---:|
| first_appear_gap | -0.007 | +0.023 |
| last_change_gap | -0.269 | -0.239 |
| max_gap | +0.332 | +0.332 |
| mean_gap | +0.100 | +0.115 |
| first_appear_step_frac | -0.141 | -0.170 |
| last_change_step_frac | -0.308 | -0.304 |
| max_gap_step_frac | +0.124 | +0.082 |
| n_valid | +0.183 | +0.164 |
| n_unique_answers | -0.277 | -0.303 |

## Notes

- This is an **offline proxy** for Phase 8. True commit-time gap (top1-top2 logit at the moment select_indices commits a token) requires a generate.py hook + GPU rerun.
- The proxy uses `parsed_answer` trajectory to define commit-equivalent steps: `first_appear` (first non-null parse), `last_change` (settlement), `max_gap` (gap-based pick).
- A single-event proxy vote that matches or beats exp_only would suggest that one commit-equivalent step carries enough signal — motivating the real hook.
- Correlation comparison (proxy vs `mean_gap`) checks whether commit-equivalent steps' gaps are more informative than the diffuse temporal mean.