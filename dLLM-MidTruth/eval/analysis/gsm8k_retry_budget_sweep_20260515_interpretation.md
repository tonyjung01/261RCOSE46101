# GSM8K T=0 Retry Budget Sweep — Interpretation

**Date**: 2026-05-15
**Companion artifacts**: `gsm8k_retry_budget_sweep_20260515.{md,json}`
**Retry artifact reused**: `outputs/.../20260514_gsm8k_retry_exp_testsubset_tau05845_v6/`
**Status**: **Monotonic growth in net rescue, no diminishing returns up to current budget k=60.** Opposite pattern from SVAMP-tuned q25 vs q40.

## Question

Is the GSM8K `tau=0.5845` budget (`k=60`, 22.73% of test) the sweet spot, or could a smaller budget capture most of the `+2.65pt` selective retry gain?

This is the budget-scaling probe that SVAMP-tuned q25 vs q40 already ran on SVAMP. On SVAMP we saw clear diminishing returns (q25 and q40 produced the same `+1.67pt` net). On GSM8K, no prior budget probe — only `k=60` was measured.

## Method

Offline only. The existing `v6` T=0 retry artifact has answers for all 60 flagged samples. For each candidate `k`, we take the bottom-`k` samples by `logistic_broad_score` (so smaller `k` = stricter score threshold), apply the existing retry on those `k` samples, keep `exp_only` on the rest, evaluate full-test accuracy.

No new GPU. Score and retry artifact held fixed; only the deployment budget varies.

## Results

| k | k/test | subset base acc | full-test acc | delta | fixes | hurts | fix/hurt |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 1.89% | 0.00% | 67.80% | `+0.38pt` | 1 | 0 | ∞ |
| 10 | 3.79% | 10.00% | 67.80% | `+0.38pt` | 2 | 1 | 2.0 |
| 15 | 5.68% | 13.33% | 67.80% | `+0.38pt` | 2 | 1 | 2.0 |
| 20 | 7.58% | 10.00% | 68.18% | `+0.76pt` | 3 | 1 | 3.0 |
| 25 | 9.47% | 16.00% | 68.56% | `+1.14pt` | 4 | 1 | 4.0 |
| 30 | 11.36% | 20.00% | 68.56% | `+1.14pt` | 4 | 1 | 4.0 |
| 40 | 15.15% | 25.00% | 68.94% | `+1.52pt` | 5 | 1 | 5.0 |
| 50 | 18.94% | 28.00% | 69.32% | `+1.89pt` | 6 | 1 | 6.0 |
| **60** | **22.73%** | **30.00%** | **70.08%** | **`+2.65pt`** | **8** | **1** | **8.0** |

## Findings

### 1. Monotonic growth in net rescue up to k=60

Fixes count grows monotonically (`1 → 2 → 2 → 3 → 4 → 4 → 5 → 6 → 8`). No plateau within the explored range. Each additional ~10 samples of budget brings ~1 additional fix on average.

### 2. Hurts stay flat at 1 across budgets ≥ 10

A single sample (`727`) contributes the only hurt; it shows up at `k=10` and stays. No additional hurts as the budget widens.

This is structurally different from SVAMP, where the q25 → q40 budget expansion added both `+1` fix and `+1` hurt. On GSM8K, expanding the budget adds fixes without adding hurts.

### 3. Fix/hurt ratio improves with k

`k=10`: 2.0 → `k=60`: 8.0. The retry policy gets *cleaner*, not noisier, as the budget widens. Wider budget on GSM8K is more discriminating, not less.

### 4. The 22.73% budget is well-tuned but probably under-budgeted

Net gain at `k=60` is the largest measured (`+2.65pt`) with no saturation signal. The trend suggests `k=80` or `k=100` might add further gain — but we cannot test this without GPU runs on score-borderline samples just outside the current flagged 60.

What we *can* say: within the available retry artifact, `k=60` is on the strict monotone side of the curve.

## Cross-task comparison

| Task | Budget scaling | Pattern |
|---|---|---|
| GSM8K (this pass) | `k=10 → k=60` | fixes `2 → 8`, hurts `1 → 1`, **fix/hurt ratio 2.0 → 8.0** |
| SVAMP-tuned | q25 → q40 (`n=14 → 23`) | fixes `1 → 2`, hurts `0 → 1`, **fix/hurt ratio ∞ → 2.0** |

**Opposite directions**:

- **GSM8K**: budget expansion is "clean" — adds fixes without hurts. The score-sorted flagged set is uniformly retry-rescuable on this task.
- **SVAMP-tuned**: budget expansion is "messy" — adds fixes and hurts in equal measure. The score-sorted flagged set saturates fast, with confident-borderline samples mixed in at moderate scores.

## Mechanism reading

Why the asymmetry? Plausible structural explanation:

- GSM8K test base acc `67.42%` → flagged 60 base `30%` → unflagged 204 base `78.43%`. The gradient from "very hard" to "very easy" samples is gradual. Many test samples are moderately hard and benefit similarly from retry.
- SVAMP test base `86.67%` → flagged 14 base `57%` → unflagged 46 base `~96–100%`. The gradient is sharp. Easy samples are saturated; widening the tau quickly pulls in samples too easy to gain from retry but easy enough to be flipped by regeneration noise.

So: **the same reliability score interacts differently with task structure**. On a task where base acc is uniformly moderate (GSM8K), score-targeted retry has a wide useful budget. On a task where most samples are confidently correct (SVAMP), the useful budget is tight.

This is consistent with all the prior findings — score targets *difficulty*; whether the difficulty translates to retry-rescuability depends on the task.

## Implications for the operational rule

- The current GSM8K selective retry budget (`tau=0.5845`, 22.73% of test) is **not** over-budgeted; if anything it is under-budgeted on this single retry artifact, with no observed saturation point.
- On SVAMP, the equivalent operating point is the **tightest budget** (q25). Wider tau hurts.
- Both readings are artifact-level, single seed.

The earlier SVAMP q25→q40 finding was sometimes summarized as "retry has diminishing returns past a tight tau." This sweep shows that summary is task-specific. On GSM8K, "retry has not yet saturated at k=60."

## Caveats

- Single retry artifact (v6, T=0). The hurt-stays-at-1 finding is single-seed; another retry seed could shift the hurt count.
- Cannot probe `k > 60` without new GPU runs on samples currently above tau.
- `k=5` has 0 hurts but only 1 fix; the 5 samples there happen to be ones where retry either fixed or did nothing. With a different starting tail of 5 samples, hurts could appear earlier.
- Subset base acc varies non-monotonically with `k` (e.g. `0%` at k=5, `10%` at k=10, `13%` at k=15, `10%` at k=20). This is because the 5 lowest-score samples happen to all be wrong; adding a few more samples pulls in some correct ones; then the next set is wrong again. Small-N artifact.

## What this adds to the project narrative

The "selective retry works on GSM8K" finding now has a richer shape:

- Original: `tau=0.5845` (`k=60`) gives `+2.65pt`
- This sweep: **gain scales smoothly with budget; +0.38pt at the lowest score tail; clean monotone growth to +2.65pt at k=60**
- Implication: the retry mechanism is not concentrated at the extreme tail (unlike SVAMP); it is broadly applicable across the flagged set
- Cross-task structural finding: same score, different budget-scaling behavior depending on the task's base-accuracy distribution

## Recommended next move

The budget-sweep gave a clean structural finding. The natural next step is the **writeup / summary consolidation** the user proposed: organize what survived (selective retry, targeting marginal, this budget sweep) vs what died (gap voting / Phase 7-family) vs what is separate (K-pool self-consistency probe).

Doing the writeup now captures the GSM8K vs SVAMP budget contrast cleanly while everything is fresh.
