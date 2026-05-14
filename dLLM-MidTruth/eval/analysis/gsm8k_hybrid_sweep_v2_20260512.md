# GSM8K Hybrid Sweep v2 — 20260512

Source: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260429_everpass_debug_bs4/gsm8k_gen128_steps64_vote_confidence_gap_answer_window5_mean_rawsum_bs4_debug/rank_0_generations.json`
Transfer: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260429_everpass_debug_bs4/svamp_gen128_steps64_vote_confidence_gap_answer_window5_mean_rawsum_bs4_debug/rank_0_generations.json`
Alpha for exp baseline: 5.0; seed=42; validation_frac=0.8; mode=phase7b

## Sanity check
- Reproduced cgap_rawsum on full GSM8K: 69.83% (tracker number for logit cgap vote: 69.83)

## Validation baselines

| Method | Val Acc |
|---|---:|
| exp_only | 70.62% |
| cgap_rawsum (logit) | 70.05% |

## Factor 0 — Exp scaling mode (shape=clipped, λ=1.0, additive)

| Exp mode | Val Acc |
|---|---:|
| raw | 70.62% |
| max_norm | 70.71% ← |
| step_frac | 70.24% |

Best exp mode: **max_norm**

## Factor 1 — wider λ sweep (shape=clipped, exp_mode=max_norm, additive)

| λ | Val Acc |
|---|---:|
| 0.25 | 70.71% |
| 0.5 | 70.71% |
| 1.0 | 70.71% |
| 2.0 | 70.81% ← |
| 5.0 | 70.62% |
| 10.0 | 70.62% |
| 25.0 | 70.71% |
| 50.0 | 70.71% |
| 100.0 | 70.71% |

Best λ: **2.0**

## Factor 2 — Formula (shape=clipped, exp_mode=max_norm, λ=2.0)

| Formula | Val Acc |
|---|---:|
| additive | 70.81% |
| multiplicative | 71.00% ← |

Best formula: **multiplicative**

## Best config

`(exp_mode, quality_shape, λ, formula) = (max_norm, clipped, 2.0, multiplicative)`

## Test (one-shot)

| Method | Test Acc |
|---|---:|
| exp_only | 67.42% |
| cgap_rawsum (logit) | 68.94% |
| hybrid(best) | 67.42% |

## Transfer check — SVAMP (full, unpatched logit cgap data)

| Method | SVAMP Acc |
|---|---:|
| exp_only | 86.33% |
| cgap_rawsum (logit) | 86.33% |
| hybrid(best) | 86.33% |

## Notes

- This v2 pass explores whether the earlier null result was caused by raw exp scale dominating `quality × λ`.
- `raw` keeps the original exp dynamic range, `max_norm` divides exp by the sample-local max exp, and `step_frac` uses linear step fraction.
- If the best config still matches `exp_only` or stays below `cgap_rawsum`, that is evidence that simple additive/multiplicative hybrids remain weak even after scale adjustments.
