# Phase E-Retry-Control — Selective vs Random Retry — 20260514

- eval split: `test` (n=`264`)
- answer kind on retry: `exp_only`
- score key / tau: `logistic_broad_score` / `0.5845`

- selective manifest: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/retry_subset_gsm8k_test_logistic_broad_tau05845_20260514.json` (n_selected=`60`)
- selective retry artifact: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260514_gsm8k_retry_exp_testsubset_tau05845_v6/rank_0_generations.json`
- random manifest: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/random_retry_subset_test_seed123_20260514.json` (n_selected=`60`)
- random retry artifact: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260514_gsm8k_retry_exp_random60_seed123/rank_0_generations.json`
- overlap between selective and random subsets: `12` samples

## Full-test policies (single seed, current artifact)

| Policy | Acc | vs baseline | changed | fixes | hurts |
|---|---:|---:|---:|---:|---:|
| P0 baseline `exp_only` | 67.42% | — | 0 | 0 | 0 |
| P_selective (score-targeted retry) | 70.08% | `+2.65pt` | 28 | 8 | 1 |
| P_random (random-budget retry) | 67.80% | `+0.38pt` | 13 | 4 | 3 |

## Subset-local accuracy (only the retried samples)

| Subset | n | base acc | retry acc | delta |
|---|---:|---:|---:|---:|
| selective (score-flagged) | 60 | 30.00% | 41.67% | `+11.67pt` |
| random | 60 | 68.33% | 70.00% | `+1.67pt` |

Subset-local rows show where the retry budget was actually spent. The selective subset's base acc shows how much headroom the score-flagged slice had; the random subset's base acc should be close to overall test accuracy. Comparing the two `delta`s shows whether retry produced more lift on the score-flagged slice than on a random slice of the same size.

## Targeting marginal value

`(P_selective − P_random) on full test = +2.27pt`

Interpretation guide (single seed, single retry artifact per policy):

- if `P_selective − P_random > 0` and the subset-local delta on the selective slice is materially larger than on the random slice, the reliability score is doing real targeting for retry budget allocation on this artifact; this is the cleanest first-look evidence we have so far.
- if `P_selective − P_random ≈ 0`, retry helps in general but the score is no better than random at choosing where to spend the budget on this artifact.
- if `P_selective − P_random < 0`, on this artifact the score is misallocating budget.
- single-seed measurement: magnitudes are noisy and the sign itself is provisional. The control still gives the cleanest first read on targeting value, but a clean operational claim still wants multi-seed and/or a complement-only random control.

## Overlap caveat

Pure random sampling can overlap with the score-flagged set. Any samples in both manifests are evaluated under each policy independently (i.e., the same sample can be a retry target for both selective and random), so the comparison stays apples-to-apples on the policy effect.
A stricter optional follow-up is a complement-only random control, drawing the random subset from the score-unflagged samples only (`select_random_retry_subset.py --exclude-flagged`); that disentangles 'score targeting' from 'random sample happened to land on flagged samples.'

## Fix/hurt sample sets

- selective fixes: `[124, 200, 411, 451, 471, 538, 646, 1093]`
- selective hurts: `[727]`
- random fixes:    `[175, 514, 928, 1059]`
- random hurts:    `[191, 663, 1146]`