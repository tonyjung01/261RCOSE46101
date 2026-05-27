# Retry Pool Re-vote Evaluation — K=3 — 20260515

- Aggregator: pool valid step events from the first `K` retry artifacts and re-run the original `exp_only` vote over the merged event set.
- flagged subset on test: `60` samples

## Per-K pooled exp re-vote on flagged subset

| K | pooled exp acc | new rescues at K | cumulative rescues |
|---:|---:|---:|---:|
| 1 | 40.00% | 14 | 14 |
| 2 | 35.00% | 0 | 14 |
| 3 | 35.00% | 2 | 16 |

## Full-test deploy: pooled exp re-vote(K) on flagged, base elsewhere

| K | full-test acc |
|---:|---:|
| 1 | 70.08% |
| 2 | 68.56% |
| 3 | 68.56% |
