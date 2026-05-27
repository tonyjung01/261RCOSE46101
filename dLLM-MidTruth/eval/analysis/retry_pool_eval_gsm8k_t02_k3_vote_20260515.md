# Phase E-Retry-Pool — K=3 — 20260515

- K = `3` retry artifacts, answer_kind=`vote`
- flagged subset on test: `60` samples

## Answer diversity sanity (vs first artifact)

| artifact | answer equal with ref | answer diff with ref |
|---|---:|---:|
| `20260515_gsm8k_retry_exp_pool_t02_seed42` | 60 | 0 |
| `20260515_gsm8k_retry_exp_pool_t02_seed43` | 11 | 49 |
| `20260515_gsm8k_retry_exp_pool_t02_seed44` | 15 | 45 |

This sanity block is answer-level, not raw-text-level: these retry artifacts do not preserve enough generation text to safely compare trajectories directly. If answer-diff with ref is `0` for all non-ref artifacts, the additional runs did not contribute any new deployed answers under the selected answer_kind, even if hidden raw trajectories differed.

## Per-K cumulative aggregates on flagged subset

| K | union-any acc | majority acc | new rescues at K | cumulative rescues |
|---:|---:|---:|---:|---:|
| 1 | 40.00% | 40.00% | 14 | 14 |
| 2 | 46.67% | 40.00% | 1 | 15 |
| 3 | 58.33% | 43.33% | 6 | 21 |

- `union-any` is the oracle ceiling at K: correct under at least one of the first K reads.
- `majority` is a deployable aggregator: majority answer among the first K reads.
- `new rescues at K` counts samples that the first `K-1` reads did not rescue but the `K`-th did. Marginal contribution.

## Full-test accuracy with majority(K) on flagged, base elsewhere

| K | full-test acc |
|---:|---:|
| 1 | 70.08% |
| 2 | 69.70% |
| 3 | 70.45% |

This is the deployable policy: vote across K independent regenerations on the score-flagged slice, keep `exp_only` on the unflagged. If full-test acc keeps growing with K, additional regeneration seeds keep paying off; if it plateaus, the diversity is saturated.

## Caveats

- single outer seed (the score-flagged set comes from seed=42 score fit).
- the answer-diversity sanity rows do **not** prove raw trajectory diversity. They only tell us whether the additional artifacts contribute different deployed answers under the selected answer_kind.