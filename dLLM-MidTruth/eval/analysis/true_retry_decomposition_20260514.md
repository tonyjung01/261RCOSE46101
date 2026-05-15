# True Retry Fix Decomposition — 20260514

- score key: `logistic_broad_score`, tau: `0.5845`
- flagged sample count: `60 / 264` (test split, seed=42)
- retry_exp artifact: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260514_gsm8k_retry_exp_testsubset_tau05845_v6/rank_0_generations.json`
- retry_prob artifact: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260514_gsm8k_retry_prob_testsubset_tau05845_v1/rank_0_generations.json`

## Accuracies on the flagged 60 samples

| Source | Correct / 60 | Acc |
|---|---:|---:|
| base | 18 | 30.00% |
| prob | 22 | 36.67% |
| retry_exp | 25 | 41.67% |
| retry_exp_vote | 25 | 41.67% |
| retry_prob | 25 | 41.67% |
| retry_prob_vote | 25 | 41.67% |

These are the per-sample accuracies on just the bottom-22.73% subset, before mixing back the unflagged 204.

## Fix / hurt count vs base (`exp_only`) on flagged 60

| Source | fixes | hurts | both correct | both wrong |
|---|---:|---:|---:|---:|
| prob_correct | 4 | 0 | 18 | 38 |
| retry_exp_correct | 8 | 1 | 17 | 34 |
| retry_exp_vote_correct | 8 | 1 | 17 | 34 |
| retry_prob_correct | 8 | 1 | 17 | 34 |
| retry_prob_vote_correct | 8 | 1 | 17 | 34 |

## Fix overlap — offline prob fallback vs true retry (exp on retry)

- offline fallback fixes (P1): `4` samples
- true retry fixes (P2):       `8` samples
- shared fixes:                `4`
- retry-only fixes:            `4`
- offline-only fixes:          `0`

Hurts:
- offline fallback hurts: `0`
- true retry hurts:       `1` (sample(s): `[727]`)

## Diversity gain on the flagged 60

- union of (base correct, prob correct, retry_exp correct): `26/60` (43.33%)
- intersect (all three correct): `17/60` (28.33%)
- **uniquely retry-rescued** (base wrong AND prob wrong AND retry right): `4/60` (6.67%)

`uniquely retry-rescued` on this artifact are samples where neither of the two available offline answer sources (`exp_only`, `prob_vote`) was right but regeneration produced the correct answer. They are the part of the retry advantage that the available offline fallback pool does not capture on this artifact; a richer offline pool could shrink this slice.

## Full-test sanity check (n=264)

| Policy | Acc | vs base |
|---|---:|---:|
| base `exp_only` | 67.42% | — |
| P1 offline fallback | 68.94% | `+1.52pt` |
| P2 true retry (retry exp_only on flagged) | 70.08% | `+2.65pt` |

This reproduces the published P1/P2 numbers from `true_retry_eval_gsm8k_exp_test_tau05845_20260514_v6.md` and serves as a cross-tab consistency check.

## Reading guide

- `retry-only fixes` count samples where retry helped but the available offline fallback sources in this artifact did not. A larger ratio over `shared fixes` indicates that, on this artifact, retry captures a slice the offline pool does not.
- `uniquely retry-rescued > 0` means the available offline pool here (`exp_only`, `prob_vote`) cannot rescue some samples that regeneration does. A richer offline pool (more seeds, more voting variants) could narrow this slice; this is an artifact-level finding, not a structural one.
- `retry hurt > offline hurt` is the single-seed regeneration-variance cost on this artifact. The 1 hurt observed here is a per-seed observation; multi-seed retry would refine the rate.
- The numbers above are single-seed (seed=42 test split) and a single retry artifact per voting method. The qualitative pattern is informative but the magnitudes should not be quoted as constants.