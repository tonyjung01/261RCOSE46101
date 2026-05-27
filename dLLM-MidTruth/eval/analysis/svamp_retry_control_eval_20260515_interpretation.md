# SVAMP Cross-task Retry Control — Interpretation

**Date**: 2026-05-15
**Companion artifacts**: `svamp_retry_control_eval_20260515.{md,json}`
**Selective manifest**: `svamp_retry_subset_logistic_broad_score_tau05845_20260515.{md,json,txt}`
**Random manifest**: `svamp_random_retry_subset_complement_seed124_20260515.{md,json,txt}`
**Retry artifacts**:
- selective: `outputs/.../20260515_svamp_retry_exp_selective_tau05845/`
- random complement: `outputs/.../20260515_svamp_retry_exp_complement_seed124/`

**Status**: **Mixed cross-task transfer.** "Random retry on confident samples is net-negative" transfers cleanly to SVAMP. "Selective retry on flagged samples produces a lift" does **not** transfer in any meaningful way: SVAMP selective subset retry delta is exactly `+0.00pt`.

## What this pass tested

Apply the GSM8K-val-fitted reliability score and `tau=0.5845` to SVAMP without any SVAMP-specific tuning. Run two GPU retry runs at the same budget (`49` samples each):

- **selective**: SVAMP samples where the score falls `≤ tau` (n=49, flagged base acc `55.10%`)
- **random complement** (seed=124, `--exclude-flagged`): a budget-matched random subset drawn from the unflagged 251 only (overlap with flagged = 0, base acc `93.88%`)

Then compare both against the SVAMP `exp_only` baseline (`86.33%`).

## Headline numbers (single seed, current SVAMP artifact)

### Full-SVAMP (n=300)

| Policy | Acc | vs base | changed | fixes | hurts |
|---|---:|---:|---:|---:|---:|
| P0 baseline `exp_only` | 86.33% | — | 0 | 0 | 0 |
| P_selective (score-targeted retry) | **86.33%** | **`+0.00pt`** | 9 | 3 | 3 |
| P_random (complement, seed=124) | **85.67%** | **`−0.67pt`** | 2 | 0 | 2 |

Targeting marginal `(P_selective − P_random) = +0.67pt` on this artifact.

### Subset-local (retried slice only)

| Subset | n | base acc | retry acc | delta |
|---|---:|---:|---:|---:|
| selective (score-flagged) | 49 | 55.10% | 55.10% | `+0.00pt` |
| random complement (seed=124) | 49 | 93.88% | 89.80% | `−4.08pt` |

## Cross-task comparison

| Axis | GSM8K (seed=42, multi-seed K=4 random control) | SVAMP (single seed) |
|---|---:|---:|
| overall base | 67.42% | 86.33% |
| flagged base | 30.00% | 55.10% |
| unflagged base | 78.43% | 93.88% |
| **selective subset retry delta** | **`+11.67pt`** | **`+0.00pt`** |
| random complement subset delta | `−1.67pt ± 2.72pt` | `−4.08pt` |
| **targeting marginal** | **`+3.03pt ± 0.62pt`** | **`+0.67pt`** |

## Findings

### 1. "Don't retry confident samples" transfers cleanly

Random retry on the unflagged 251 (`base 93.88%`) drops the subset to `89.80%` (`−4.08pt`, 0 fixes, 2 hurts on 49 samples). That is a sharper version of the same pattern observed on GSM8K (`−1.67pt ± 2.72pt` across 4 random-control seeds): regeneration variance on high-confidence samples flips correct answers more often than it fixes wrong ones, and SVAMP's higher base accuracy on the unflagged set makes the effect more pronounced.

This part of the targeting story — the reliability score correctly **avoids** the retry-bad region — generalizes to SVAMP on this single-seed artifact.

### 2. "Selective retry on flagged samples produces a lift" does NOT transfer

The SVAMP selective subset retry produced exactly `3 fixes` and `3 hurts` on 49 samples — net `+0.00pt`. By contrast the GSM8K selective subset retry produced `8 fixes / 1 hurt` for `+11.67pt`. Same score, same threshold, same retry pipeline; the lift collapses entirely on SVAMP.

The flagged 49 on SVAMP are clearly harder than the overall pool (`55.10%` vs `86.33%` base), so the score is doing some difficulty targeting. But fresh regeneration on these specific hard SVAMP samples does not move accuracy — fixes are balanced by hurts.

### 3. Plausible explanation

The GSM8K Phase E-Diff result showed `Pearson(score, delta=y_prob−y_exp) ≈ 0`: the reliability score predicts *difficulty*, not *method-switch utility*. On GSM8K, "difficult" samples happened to also be retry-rescuable (regeneration variance produced fixes net of hurts), so the score's difficulty targeting indirectly drove retry value. On SVAMP, that coincidence breaks: the flagged samples are difficult, but the difficulty is not the retry-rescuable kind — perhaps the model genuinely cannot solve them, so fresh trajectories give a uniform distribution of wrong-but-different answers rather than a concentrated set of fixes.

This is consistent with the rest of the project's framing: the score is a *difficulty* signal, not a guarantee of retry utility. On GSM8K those two correlate; on SVAMP they decorrelate.

### 4. The targeting marginal `+0.67pt` is mostly driven by the random control hurting

`+0.67pt = +0.00pt selective − (−0.67pt random) = +0.67pt`. The marginal is positive only because the random control loses ground; selective itself does not produce a lift. On `n=300` SVAMP, `+0.67pt` corresponds to `2` samples — at the boundary of what we should call signal vs noise on this scale.

## What transfers / what doesn't

| Component | Transfers to SVAMP? |
|---|---|
| Score targets the "difficult" slice (lower base acc) | ✓ (flagged 55% vs unflagged 94%) |
| Retry on unflagged samples is net-negative | ✓ (sharper on SVAMP, `−4.08pt` vs GSM8K `−1.67pt`) |
| Selective retry on flagged samples produces a lift | ✗ (`+0.00pt` on SVAMP) |
| Targeting marginal is a meaningful operational gain | ✗ on this single seed (`+0.67pt ≈ 2 samples on n=300`) |

## Caveats

- single SVAMP retry per policy; no multi-seed yet for SVAMP
- the `3 fixes / 3 hurts` exact balance on selective is small-sample (n=49) and another draw could push it +1 or −1 either way
- the SVAMP score-fitting reuses the GSM8K-val-fitted logistic_broad without any SVAMP-specific recalibration; a SVAMP-tuned score might behave differently
- the comparison artifact for GSM8K (`v6` selective + 4 complement seeds + pure-random seed=123) is much richer than the SVAMP single-pair comparison; magnitude comparisons across tasks should be read with that asymmetry in mind

## Recommended next move

The clean conclusion at this stage is: **the GSM8K selective-retry rule, as is, does not produce a SVAMP lift.** Options to refine, in rough order:

1. **SVAMP multi-seed random control** (`seed=125, 126, 127`, `--exclude-flagged`): puts variance bars on the `−4.08pt` random subset delta and the `+0.67pt` targeting marginal. Cheap. Helps distinguish "SVAMP selective is exactly zero" from "SVAMP selective is small but real."
2. **SVAMP-tuned score**: refit the reliability score on a SVAMP val split (instead of transferring GSM8K). Tests whether SVAMP's difficulty signal even exists in the same feature space, and whether a SVAMP-fit `tau` would isolate retry-rescuable samples better.
3. **SVAMP score+retry decomposition** (Phase E-Retry-Decomp on SVAMP): check whether the 3 SVAMP selective fixes overlap with offline prob_vote fixes on the same samples — i.e., is the selective retry on SVAMP producing **any** unique-to-retry rescue, or are all 3 fixes already in the offline pool?
4. **Math500 cross-task**: same exercise on Math500 (base `~24%`, very different regime). Likely an even more extreme failure mode given Math500's high AWNF rate.

(1) is the cheapest and gives the cleanest follow-up read. (2) is more ambitious. (3) is offline-only and a quick mechanism check.
