# Reliability-aware Abstention (Phase E) — 20260513

Source: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260429_everpass_debug_bs4/gsm8k_gen128_steps64_vote_confidence_gap_answer_window5_mean_rawsum_bs4_debug/rank_0_generations.json`
Transfer: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260429_everpass_debug_bs4/svamp_gen128_steps64_vote_confidence_gap_answer_window5_mean_rawsum_bs4_debug/rank_0_generations.json`
seed=42; validation_frac=0.8

GSM8K n=1319 (val 1055 / test 264); SVAMP n=300

Base accuracy (no abstention):
- GSM8K full: 69.98%
- GSM8K val:  70.62%
- GSM8K test: 67.42%
- SVAMP:      86.33%

## Single-feature val Pearson

| Feature | Val Pearson |
|---|---:|
| max_gap | +0.337 |
| last_change_step_frac | -0.308 |
| last_change_gap | -0.264 |
| n_unique_answers | -0.263 |
| std_gap | +0.239 |
| n_valid | +0.182 |
| first_appear_step_frac | -0.158 |
| max_gap_step_frac | +0.116 |
| mean_gap | +0.095 |
| first_appear_gap | -0.024 |

Combined score features (kept |corr|>=0.05): ['max_gap', 'mean_gap', 'last_change_gap', 'std_gap', 'n_valid', 'n_unique_answers', 'last_change_step_frac', 'max_gap_step_frac', 'first_appear_step_frac']

## GSM8K Validation (n=1055)

| Score | AURC | SelAcc@80% | SelAcc@50% | Cov@90%acc |
|---|---:|---:|---:|---:|
| max_gap | 0.2309 | 76.54% | 83.71% | n/a |
| last_change_gap | 0.2440 | 76.78% | 78.41% | n/a |
| n_unique_answers | 0.2449 | 74.64% | 82.58% | n/a |
| last_change_step_frac | 0.2475 | 73.58% | 83.14% | n/a |
| std_gap | 0.2521 | 74.05% | 79.92% | n/a |
| first_appear_step_frac | 0.2605 | 74.53% | 76.14% | n/a |
| n_valid | 0.2656 | 73.10% | 76.33% | n/a |
| mean_gap | 0.2685 | 72.51% | 75.76% | n/a |
| max_gap_step_frac | 0.2796 | 72.04% | 72.73% | n/a |
| first_appear_gap | 0.3025 | 69.19% | 69.70% | n/a |
| **combined** | **0.2187** | **78.08%** | **86.74%** | **n/a** |
| oracle | 0.1234 | 88.27% | 100.00% | — |
| random | 0.2985 | 69.55% | 70.08% | — |

## GSM8K Test (one-shot) (n=264)

| Score | AURC | SelAcc@80% | SelAcc@50% | Cov@90%acc |
|---|---:|---:|---:|---:|
| max_gap | 0.2626 | 72.99% | 79.55% | n/a |
| n_unique_answers | 0.2636 | 72.99% | 80.30% | n/a |
| last_change_gap | 0.2689 | 73.46% | 75.76% | n/a |
| last_change_step_frac | 0.2735 | 71.56% | 81.82% | n/a |
| n_valid | 0.2937 | 71.56% | 71.21% | n/a |
| std_gap | 0.2941 | 70.14% | 75.00% | n/a |
| max_gap_step_frac | 0.2998 | 70.62% | 71.97% | n/a |
| first_appear_step_frac | 0.3120 | 69.19% | 68.94% | n/a |
| mean_gap | 0.3128 | 68.72% | 71.21% | n/a |
| first_appear_gap | 0.3470 | 65.88% | 64.39% | n/a |
| **combined** | **0.2515** | **74.88%** | **81.06%** | **n/a** |
| oracle | 0.1521 | 84.36% | 100.00% | — |
| random | 0.3034 | 70.62% | 69.70% | — |

## SVAMP transfer (n=300)

| Score | AURC | SelAcc@80% | SelAcc@50% | Cov@90%acc |
|---|---:|---:|---:|---:|
| n_unique_answers | 0.0816 | 92.50% | 96.00% | 85% |
| last_change_gap | 0.0855 | 91.25% | 95.33% | 90% |
| max_gap | 0.0898 | 90.83% | 95.33% | 80% |
| last_change_step_frac | 0.0974 | 90.42% | 94.00% | 80% |
| std_gap | 0.1083 | 89.58% | 90.67% | 60% |
| first_appear_gap | 0.1178 | 87.50% | 90.67% | 70% |
| n_valid | 0.1234 | 88.33% | 88.00% | n/a |
| mean_gap | 0.1274 | 86.67% | 89.33% | n/a |
| first_appear_step_frac | 0.1363 | 86.25% | 84.00% | n/a |
| max_gap_step_frac | 0.1475 | 85.00% | 84.00% | n/a |
| **combined** | **0.0954** | **91.67%** | **91.33%** | **90%** |
| oracle | 0.0298 | 100.00% | 100.00% | — |
| random | 0.1366 | 86.25% | 86.67% | — |

## Reading guide

- **AURC**: mean (1 − selective accuracy) across the coverage sweep. Lower = better. `random` and `oracle` bracket the achievable range; a score is only useful if its AURC sits meaningfully closer to oracle than to random.
- **SelAcc@80%**: selective accuracy when we keep the 80% of samples most reliable by the score (abstain on the bottom 20%). A useful score should lift this above the base accuracy at the same coverage.
- **Cov@90%acc**: the largest coverage at which selective accuracy stays ≥ 90%. `n/a` means no coverage point in the sweep crosses 90% selective accuracy.
- Combined score is the val-fitted signed z-score sum, applied unchanged to test and SVAMP — no test/transfer-side refitting.
