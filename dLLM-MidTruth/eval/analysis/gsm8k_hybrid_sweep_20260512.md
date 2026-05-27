# GSM8K Hybrid One-Factor Sweep — 20260512

Source: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260429_everpass_debug_bs4/gsm8k_gen128_steps64_vote_confidence_gap_answer_window5_mean_rawsum_bs4_debug/rank_0_generations.json`
Transfer: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260429_everpass_debug_bs4/svamp_gen128_steps64_vote_confidence_gap_answer_window5_mean_rawsum_bs4_debug/rank_0_generations.json`
Alpha for exp baseline: 5.0; seed=42; validation_frac=0.8

## Sanity check
- Reproduced cgap_rawsum on full GSM8K: 69.83% (tracker number for logit cgap vote: 69.83)

## Validation baselines

| Method | Val Acc |
|---|---:|
| exp_only | 70.62% |
| cgap_rawsum (logit) | 70.05% |

## Factor 1 — Quality shape (λ=1.0, additive)

| Shape | Val Acc |
|---|---:|
| binary | 70.52% |
| clipped | 70.62% ← |
| tanh | 70.62% |

Best shape: **clipped**

## Factor 2 — λ sweep (shape=clipped, additive)

| λ | Val Acc |
|---|---:|
| 0.25 | 70.62% ← |
| 0.5 | 70.62% |
| 1.0 | 70.62% |
| 2.0 | 70.62% |
| 5.0 | 70.62% |

Best λ: **0.25**

## Factor 3 — Formula (shape=clipped, λ=0.25)

| Formula | Val Acc |
|---|---:|
| additive | 70.62% |
| multiplicative | 70.71% ← |

Best formula: **multiplicative**

## Best config

`(quality_shape, λ, formula) = (clipped, 0.25, multiplicative)`

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
