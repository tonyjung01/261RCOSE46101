# GSM8K Gap-as-Filter Sweep — 20260513

Source: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260429_everpass_debug_bs4/gsm8k_gen128_steps64_vote_confidence_gap_answer_window5_mean_rawsum_bs4_debug/rank_0_generations.json`
Transfer: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260429_everpass_debug_bs4/svamp_gen128_steps64_vote_confidence_gap_answer_window5_mean_rawsum_bs4_debug/rank_0_generations.json`
Alpha for exp baseline: 5.0; seed=42; validation_frac=0.8

## Validation baselines

| Method | Val Acc |
|---|---:|
| exp_only | 70.62% |
| cgap_rawsum | 70.05% |

## Validation sweep

| Quantile | Gap-filter Val Acc | Late-control Val Acc | Mean keep ratio |
|---:|---:|---:|---:|
| 0 | 70.62% | 70.62% | 100.0% |
| 25 | 70.43% | 70.71% | 74.8% |
| 50 | 70.52% | 70.71% | 50.6% |
| 75 | 70.62% | 70.71% | 25.8% |

Best quantile by validation: **0**

## Test (one-shot, best validation quantile)

| Method | Test Acc | No-vote |
|---|---:|---:|
| exp_only | 67.42% | 0 |
| cgap_rawsum | 68.94% | 0 |
| gap_filter(q=0) | 67.42% | 0 |
| late_control(q=0) | 67.42% | 0 |

## Transfer — SVAMP (same quantile)

| Method | SVAMP Acc |
|---|---:|
| exp_only | 86.33% |
| cgap_rawsum | 86.33% |
| gap_filter(q=0) | 86.33% |
| late_control(q=0) | 86.33% |

## Notes

- Gap-filter uses gap only to prune events, then reverts to the original exp temporal accumulation on the survivors.
- Late-control keeps the same number of latest events per sample, which helps check whether any lift is due to gap ranking rather than simple late-step pruning.
- If gap-filter outperforms both exp_only and matched-count late-control, that is evidence that the gap ranking is more actionable as a gate than as a direct sum weight.
