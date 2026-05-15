# Complement-only Random Control — Interpretation

**Date**: 2026-05-14
**Companion artifacts**: `retry_control_complement_eval_20260514.{md,json}`
**Manifest**: `random_retry_subset_test_complement_seed124_20260514.{md,json,txt}`
**Retry artifact**: `outputs/.../20260514_gsm8k_retry_exp_complement_seed124/`
**Status**: Stricter single-seed control. **Targeting marginal value increases** from `+2.27pt` (pure-random seed=123, overlap 12/60) to `+3.03pt` (complement seed=124, overlap 0/60). On this artifact, the score-targeting claim is at least as strong under the stricter control.

## What this pass adds

The pure-random control (seed=123) drew the random 60 from the full test pool and happened to overlap with the score-flagged set on `12/60` samples. That overlap potentially leaked some of the targeting effect into the random control. The complement control draws the random 60 from the **unflagged 204 only**, giving `0/60` overlap. It is the cleanest single-seed answer to "does the score know how to spend retry budget better than picking any other 60 samples from the population?"

## Headline numbers (gsm8k_test n=264, current artifact, seed=42 outer split)

| Policy | Acc | vs base | changed | fixes | hurts |
|---|---:|---:|---:|---:|---:|
| P0 baseline `exp_only` | 67.42% | — | 0 | 0 | 0 |
| P_selective (score-targeted retry) | 70.08% | `+2.65pt` | 28 | 8 | 1 |
| P_random pure (seed=123, overlap 12/60) | 67.80% | `+0.38pt` | 13 | 4 | 3 |
| **P_random complement (seed=124, overlap 0/60)** | **67.05%** | **`−0.38pt`** | 6 | 2 | 3 |

**Targeting marginal under complement control: `P_selective − P_random_complement = +3.03pt`** (vs `+2.27pt` under pure-random).

## Subset-local accuracy on the 60 retried samples

| Subset | n | base | retry | delta |
|---|---:|---:|---:|---:|
| selective (flagged) | 60 | 30.00% | 41.67% | `+11.67pt` |
| complement random (seed=124) | 60 | 76.67% | 75.00% | `−1.67pt` |

## Findings

### 1. Stricter control gives **larger** targeting marginal

The pure-random control was actually slightly **deflated** relative to the complement control: removing the 12 flagged samples from the random side made the random side worse (`67.80% → 67.05%`, i.e. `−0.75pt`), and the targeting marginal grew correspondingly (`+2.27pt → +3.03pt`, also `+0.76pt`). This is the right direction: when we remove the score-flagged contamination from the random control, the targeting claim does not weaken, it strengthens by the amount of that contamination on this artifact.

This is a single-seed observation and the `+0.76pt` shift is `~2 samples` on n=264, so the absolute magnitude is small. The directional finding — complement control does not erode the targeting claim — is the load-bearing piece.

### 2. Retry on unflagged samples is net-negative on this artifact

The complement subset's retry delta is `−1.67pt` (`2` fixes, `3` hurts on `60` samples). The unflagged 204 are already at `78.4%` base accuracy, so they have less headroom upward and more room downward; regeneration variance flips correct answers more often than it fixes wrong ones on this slice.

This is a stronger claim than just "retry helps in general — and it helps more on flagged samples." On this artifact, **retry on unflagged samples is actively harmful**: random regeneration on samples the model is already confident about is a worse policy than not retrying at all.

This is the strongest single-seed evidence we have that the reliability score is doing something specifically useful for compute allocation: it is not just picking samples where retry helps a bit; it is picking samples where retry has positive expected value, *as opposed to* the rest of the population where it has negative expected value.

### 3. Clean targeting gradient across the three controls

| Retry subset | overlap with flagged | retry delta on that subset |
|---|---:|---:|
| score-flagged (selective) | 60/60 | `+11.67pt` |
| pure random (seed=123) | 12/60 | `+1.67pt` |
| complement (seed=124) | 0/60 | `−1.67pt` |

More flagged samples in the retry budget → more positive subset-local retry delta. The monotonic gradient on this artifact is consistent with "the reliability score is picking samples that have real retry headroom; samples it does not pick do not." It is artifact-level, single-seed, and does not generalize without more seeds.

## What this does NOT establish

- **Multi-seed robustness**: the targeting marginal `+3.03pt` is a single point estimate on n=264. A 2–3 random-seed control sweep would tell us whether this lands at `+2.5 ± 0.7pt` or `+3.0 ± 0.4pt` or something with a wide CI.
- **Cross-task transfer**: this is GSM8K only. SVAMP / Math500 retry behavior may differ.
- **Cost-efficiency multiples**: we still avoid quoting fixed ratios like "Nx more efficient than uniform retry"; the artifact-level numbers are the right level of claim.
- **The `−1.67pt` complement retry delta is a single-seed observation**: it means "this particular regeneration on these particular unflagged 60 hurt 1 sample net." With multi-seed, we'd want to see whether the unflagged retry delta is robustly ≤ 0 across seeds, which would solidify "retry on unflagged is net-negative" as a defensible pattern.

## What this strengthens

- The selective-retry result (P_selective `+2.65pt`) is no longer just "retry happens to help on a small slice"; under a strict overlap-free control, it is `+3.03pt` better than retrying a same-size random slice from the unflagged complement.
- The reliability score's value is now a per-allocation-decision claim: "use compute on flagged samples, not on unflagged ones."
- The plan's reordered priority (Complement first, then multi-seed) was the right call: complement strengthened the targeting story rather than weakening it, so multi-seed is the natural next refinement.

## Recommended next step (2순위 in the priority list)

Run 2–3 additional random-seed controls. Options:

- **Complement-only**, seeds `125, 126, 127`: draws each from the unflagged 204. Gives a clean multi-seed estimate of `(P_selective − P_random_complement)` mean and std.
- **Pure-random**, seeds `125, 126, 127`: matches the original pure-random control's setup. Same multi-seed estimate but for the pure-random marginal.

Pick one of these (complement-only is slightly stricter), keep consistent, and we'll have a multi-seed CI on the targeting marginal. Cost: 2–3 GPU reruns at `~30s` each.
