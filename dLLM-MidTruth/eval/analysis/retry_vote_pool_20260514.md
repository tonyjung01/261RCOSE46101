# Retry Vote-Method Pool — 20260514

This is a **cheap analog** to Phase E-Retry-Pool: a vote-method pool over a single retry trajectory. Not a regeneration-seed pool.

- retry exp artifact: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260514_gsm8k_retry_exp_testsubset_tau05845_v6/rank_0_generations.json`
- retry prob artifact: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260514_gsm8k_retry_prob_testsubset_tau05845_v1/rank_0_generations.json`
- raw generation text equality between the two artifacts: `60/60` — the two retry artifacts share the same retry trajectory (temperature=0.0 makes generation deterministic)
- flagged subset on test: `60` samples

## Per-read accuracy on the flagged 60

| Read | Correct / 60 | Acc |
|---|---:|---:|
| base `exp_only` (before retry) | 18 | 30.00% |
| retry `v6_exp_only` | 25 | 41.67% |
| retry `v6_vote` | 25 | 41.67% |
| retry `v1_exp_only` | 25 | 41.67% |
| retry `v1_vote` | 25 | 41.67% |
| **union-any** (correct under ≥1 read) | **26** | **43.33%** |
| intersect-all (correct under all reads) | 24 | 40.00% |

`union-any − single-read` is the upper bound for vote-method diversity rescue **on this trajectory** (i.e. on this artifact). To get above this ceiling on the flagged samples, fresh regeneration trajectories are needed.

## Full-test ceiling under an oracle pool picker

If we had an oracle that picked the right read out of `{base, v6_exp, v6_vote, v1_exp, v1_vote}` on each flagged sample, full-test accuracy would be:

| Strategy | Full-test acc |
|---|---:|
| baseline | 67.42% |
| oracle pool over reads on flagged 60 | 70.83% |

This is a **ceiling**, not deployable — picking the right read requires knowing the answer. Still informative as an upper bound for what vote-method diversity could buy on this single trajectory.

## Reads where the union pool helps beyond `v6_exp_only`

`1` flagged samples are correct under at least one read but not under `v6_exp_only`. Sample indices: `[841]`.

## What this means for true E-Retry-Pool

- The two existing retry artifacts (`v6` exp, `v1` prob) share the same retry trajectory (raw generation text equal `60/60`), so they are a vote-method pool, not a regeneration-seed pool.
- For a real diversity-pool study, fresh retry runs with non-zero temperature or a different sampling seed are needed; the K-seed pool size should aggregate over different trajectories.
- The vote-method pool union here gives a partial intuition: if even just swapping vote method on a single trajectory rescues additional samples, fresh trajectories should rescue strictly more.