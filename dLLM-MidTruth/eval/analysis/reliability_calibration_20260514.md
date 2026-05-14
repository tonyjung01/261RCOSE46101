# Reliability Calibration (Phase E5) — 20260514

Binned calibration analysis for the learned `logistic_broad` reliability score.

Sources:
- GSM8K base: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260429_everpass_debug_bs4/gsm8k_gen128_steps64_vote_confidence_gap_answer_window5_mean_rawsum_bs4_debug/rank_0_generations.json`
- SVAMP transfer: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260429_everpass_debug_bs4/svamp_gen128_steps64_vote_confidence_gap_answer_window5_mean_rawsum_bs4_debug/rank_0_generations.json`
- Math500 base: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260430_cgap_answer_window5_prob_mean_rawsum_bs4_all_debug/math500_gen128_steps64_vote_confidence_gap_answer_window5_prob_mean_rawsum_bs4/rank_0_generations.json`

Interpretation notes:
- Lower ECE / MCE / Brier are better.
- SVAMP uses the GSM8K-fitted model unchanged, so this is a transfer calibration readout.
- Math500 uses its own task-specific learned score because the feature set differs materially.

## GSM8K validation

- ECE: `0.0245`
- MCE: `0.2516`
- Brier: `0.1651`

| Bin | Prob range | n | Mean predicted p | Empirical acc | |acc-p| |
|---|---|---:|---:|---:|---:|
| 0 | [0.0, 0.1) | 3 | 0.082 | 0.333 | 0.252 |
| 1 | [0.1, 0.2) | 14 | 0.168 | 0.143 | 0.025 |
| 2 | [0.2, 0.3) | 41 | 0.262 | 0.293 | 0.031 |
| 3 | [0.3, 0.4) | 49 | 0.358 | 0.347 | 0.011 |
| 4 | [0.4, 0.5) | 79 | 0.446 | 0.418 | 0.029 |
| 5 | [0.5, 0.6) | 96 | 0.554 | 0.583 | 0.029 |
| 6 | [0.6, 0.7) | 123 | 0.651 | 0.618 | 0.034 |
| 7 | [0.7, 0.8) | 210 | 0.755 | 0.729 | 0.027 |
| 8 | [0.8, 0.9) | 283 | 0.853 | 0.873 | 0.020 |
| 9 | [0.9, 1.0] | 157 | 0.927 | 0.943 | 0.016 |

## GSM8K test

- ECE: `0.0596`
- MCE: `0.2778`
- Brier: `0.1729`

| Bin | Prob range | n | Mean predicted p | Empirical acc | |acc-p| |
|---|---|---:|---:|---:|---:|
| 0 | [0.0, 0.1) | 1 | 0.098 | 0.000 | 0.098 |
| 1 | [0.1, 0.2) | 2 | 0.148 | 0.000 | 0.148 |
| 2 | [0.2, 0.3) | 7 | 0.258 | 0.143 | 0.115 |
| 3 | [0.3, 0.4) | 12 | 0.361 | 0.083 | 0.278 |
| 4 | [0.4, 0.5) | 18 | 0.456 | 0.444 | 0.011 |
| 5 | [0.5, 0.6) | 24 | 0.558 | 0.375 | 0.183 |
| 6 | [0.6, 0.7) | 26 | 0.653 | 0.692 | 0.039 |
| 7 | [0.7, 0.8) | 52 | 0.756 | 0.731 | 0.025 |
| 8 | [0.8, 0.9) | 80 | 0.855 | 0.812 | 0.043 |
| 9 | [0.9, 1.0] | 42 | 0.925 | 0.905 | 0.020 |

## SVAMP transfer

- ECE: `0.1155`
- MCE: `0.2853`
- Brier: `0.1091`

| Bin | Prob range | n | Mean predicted p | Empirical acc | |acc-p| |
|---|---|---:|---:|---:|---:|
| 0 | [0.0, 0.1) | 0 | — | — | — |
| 1 | [0.1, 0.2) | 3 | 0.142 | 0.000 | 0.142 |
| 2 | [0.2, 0.3) | 2 | 0.220 | 0.000 | 0.220 |
| 3 | [0.3, 0.4) | 14 | 0.357 | 0.357 | 0.000 |
| 4 | [0.4, 0.5) | 8 | 0.465 | 0.750 | 0.285 |
| 5 | [0.5, 0.6) | 29 | 0.556 | 0.793 | 0.237 |
| 6 | [0.6, 0.7) | 41 | 0.657 | 0.927 | 0.270 |
| 7 | [0.7, 0.8) | 37 | 0.758 | 0.838 | 0.080 |
| 8 | [0.8, 0.9) | 110 | 0.849 | 0.918 | 0.069 |
| 9 | [0.9, 1.0] | 56 | 0.929 | 0.982 | 0.053 |

## Math500 validation

- ECE: `0.0000`
- MCE: `0.0000`
- Brier: `0.1829`

| Bin | Prob range | n | Mean predicted p | Empirical acc | |acc-p| |
|---|---|---:|---:|---:|---:|
| 0 | [0.0, 0.1) | 0 | — | — | — |
| 1 | [0.1, 0.2) | 0 | — | — | — |
| 2 | [0.2, 0.3) | 400 | 0.253 | 0.253 | 0.000 |
| 3 | [0.3, 0.4) | 0 | — | — | — |
| 4 | [0.4, 0.5) | 0 | — | — | — |
| 5 | [0.5, 0.6) | 0 | — | — | — |
| 6 | [0.6, 0.7) | 0 | — | — | — |
| 7 | [0.7, 0.8) | 0 | — | — | — |
| 8 | [0.8, 0.9) | 0 | — | — | — |
| 9 | [0.9, 1.0] | 0 | — | — | — |

## Math500 test

- ECE: `0.0442`
- MCE: `0.0442`
- Brier: `0.1646`

| Bin | Prob range | n | Mean predicted p | Empirical acc | |acc-p| |
|---|---|---:|---:|---:|---:|
| 0 | [0.0, 0.1) | 0 | — | — | — |
| 1 | [0.1, 0.2) | 0 | — | — | — |
| 2 | [0.2, 0.3) | 100 | 0.254 | 0.210 | 0.044 |
| 3 | [0.3, 0.4) | 0 | — | — | — |
| 4 | [0.4, 0.5) | 0 | — | — | — |
| 5 | [0.5, 0.6) | 0 | — | — | — |
| 6 | [0.6, 0.7) | 0 | — | — | — |
| 7 | [0.7, 0.8) | 0 | — | — | — |
| 8 | [0.8, 0.9) | 0 | — | — | — |
| 9 | [0.9, 1.0] | 0 | — | — | — |

## Current read

- This pass is about whether the learned reliability score is calibrated enough to be read as a confidence-like quantity, not just as a ranking signal.
- If monotonic ranking is good but calibration is loose, threshold-based abstention/retry may still work while direct probability interpretation remains weak.
