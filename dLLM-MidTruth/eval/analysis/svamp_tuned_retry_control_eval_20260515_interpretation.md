# SVAMP-tuned Retry Control — Interpretation

**Date**: 2026-05-15
**Companion artifacts**: `svamp_tuned_retry_control_eval_20260515.{md,json}`
**Selective manifest**: `svamp_tuned_retry_subset_q25_20260515.{md,json,txt}`
**Random manifest**: `svamp_tuned_random_retry_subset_complement_seed124_q25_20260515.{md,json,txt}`
**Retry artifacts**:
- selective: `outputs/.../20260515_svamp_retry_exp_svamptuned_q25/`
- random complement: `outputs/.../20260515_svamp_retry_exp_svamptuned_complement_seed124/`
**Status**: **Weak positive on SVAMP.** Score-tuning on SVAMP val (instead of transferring from GSM8K) gives a small but positive selective retry lift on the SVAMP test split. Magnitudes are tiny — `1` fix on `14` flagged samples — but the direction reverses the mixed-transfer story: SVAMP retry is not fundamentally broken, the earlier cross-task null was partly score-precision and budget effects.

## What this pass tested

The mechanism question from the prior cross-task pass: is the SVAMP mixed-transfer result (P_selective `+0.00pt` on full SVAMP, 3 fixes / 3 hurts) because (a) the GSM8K-fitted score doesn't transfer, or (b) SVAMP "difficulty" is structurally not retry-rescuable?

This pass answers (a): refit the same `logistic_broad` pipeline on SVAMP val (80/20 outer split, seed=42), apply to SVAMP test, retry the flagged subset, compare against a complement-only random control on the same test split.

## Headline numbers (SVAMP test n=60, single seed)

| Policy | Acc | vs base | changed | fixes | hurts |
|---|---:|---:|---:|---:|---:|
| P0 baseline `exp_only` | 86.67% | — | 0 | 0 | 0 |
| P_selective (SVAMP-tuned, flagged 14) | **88.33%** | **`+1.67pt`** | 1 | 1 | 0 |
| P_random (complement, seed=124, 14 samples) | 86.67% | `+0.00pt` | 0 | 0 | 0 |

Subset-local:

| Subset | n | base acc | retry acc | delta |
|---|---:|---:|---:|---:|
| selective (SVAMP-tuned flagged) | 14 | 57.14% | 64.29% | **`+7.14pt`** |
| random complement (seed=124) | 14 | 100.00% | 100.00% | `+0.00pt` |

Targeting marginal `(P_selective − P_random) = +1.67pt`.

## Findings

### 1. SVAMP-tuned selective retry is positive — barely, but cleanly

`1 fix / 0 hurts` on the flagged 14. The 1 fix shifts the test accuracy by `1/60 = 1.67pt`. The random complement subset has base accuracy 100% (lucky draw from the unflagged 46), so the random control contributes 0 fixes and 0 hurts — random retry on this seed is a strict no-op rather than the net-negative we expect on average.

The selective `+7.14pt` subset-local delta is "`1` more correct out of `14` retried" — the magnitude is highly sample-dependent. Read directionally, not as a stable estimate.

### 2. Per-base-wrong retry rescue rate is roughly task-portable

Comparing the rate at which retry rescues base-wrong flagged samples across tasks:

| Task | retry budget | base-wrong in flagged | retry rescues | rescue rate |
|---|---:|---:|---:|---:|
| GSM8K (seed=42, selective v6) | 60 | 42 | 8 | ~19% |
| **SVAMP-tuned (this pass, q25)** | 14 | 6 | 1 | **~17%** |
| SVAMP-transferred (prior cross-task) | 49 | 22 | 3 | ~14% (with 3 hurts) |

This is a striking artifact-level observation: regeneration on a base-wrong flagged sample produces a fix at roughly comparable rates across GSM8K and SVAMP. The "SVAMP retry doesn't work" reading from the cross-task pass was partly about budget size and the resulting hurt count, not about a fundamentally different rescue rate.

We should not over-claim — `1/6` and `8/42` are both small counts — but the qualitative pattern (retry produces a small constant rate of fixes per base-wrong flagged sample) holds on both tasks.

### 3. The earlier SVAMP mixed transfer was a budget × precision effect, not a fundamental failure

Comparing the two SVAMP runs side by side:

| | GSM8K-transferred (cross-task) | SVAMP-tuned (this pass) |
|---|---:|---:|
| eval scope | full SVAMP (n=300) | SVAMP test (n=60) |
| budget | 49 | 14 |
| flagged base acc | 55.10% | 57.14% |
| selective fixes | 3 | 1 |
| selective hurts | 3 | 0 |
| selective subset retry delta | `+0.00pt` | `+7.14pt` |
| selective full-eval delta | `+0.00pt` | `+1.67pt` |

The GSM8K-transferred score's 49-sample budget picked up `3` fixes but also `3` hurts. The SVAMP-tuned score's smaller 14-sample budget produced `1` fix and `0` hurts. Reading these together, the earlier `+0.00pt` was driven by a wash between fixes and hurts at the larger budget, not by SVAMP retry being structurally null.

This is consistent but not definitive — we don't have a SVAMP-tuned retry at the same 49-sample budget to compare directly. The clean per-base-wrong rescue rate similarity (`~17%` vs `~19%`) is the strongest piece of artifact-level evidence that retry mechanism transfers across tasks.

### 4. Offline diagnostic preserved

Pre-GPU diagnostics: `prob_vote can_fix / base_wrong = 0/6` on the SVAMP-tuned flagged 14. Post-GPU: retry produced `1` fix among those 6 base-wrong samples. That 1 fix is in the **uniquely retry-rescued** slice — no static offline answer source contains the correct answer for that sample, but regeneration found it. This matches the GSM8K Phase E-Retry-Decomp pattern (`4/60` uniquely retry-rescued on GSM8K).

## Caveats

- SVAMP test n=60 is small; flagged budget n=14 is smaller. A 1-sample fix flip changes the headline.
- Single outer split (seed=42), single retry per policy, single random-control seed.
- The random complement subset happened to have base acc 100% on this seed. With seeds where the unflagged complement is below 100%, random retry would have some hurt opportunities and the targeting marginal could shift.
- The SVAMP-tuned score is fit on the same outer split as the selective subset (only the val half), so there is no leakage on test, but the `n_val = 240` is itself small for fitting 9 features.
- We do not have multi-seed SVAMP-tuned random controls yet; the `+1.67pt` marginal is single-seed.

## Synthesis with the prior SVAMP cross-task pass

| Claim | Cross-task only | + SVAMP-tuned |
|---|---|---|
| "Score targets difficulty on SVAMP" | ✓ (flagged 55% vs unflagged 94%) | ✓ confirmed (57% vs 96% on test) |
| "Retry on confident SVAMP samples is net-negative" | ✓ (`−4.08pt` complement subset) | inconclusive on this run (random complement happened to be all-correct, retry deterministic so no change) |
| "Selective retry on SVAMP produces a lift" | ✗ (`+0.00pt`) | weakly ✓ (`+1.67pt` full-test, 1 fix) |
| "Retry rescue rate is task-portable" | n/a | weakly ✓ (~17% on SVAMP vs ~19% on GSM8K) |

The clean conclusion at this point: **SVAMP-side retry mechanism is not fundamentally broken**. Score-tuning on SVAMP val gives a small positive lift on the SVAMP test split. The earlier mixed-transfer narrative was partly an artifact of budget size and score-fit imprecision when transferring across tasks.

## Recommended next step

The two open questions worth a follow-up:

1. **SVAMP-tuned at larger budget**: re-run SVAMP-tuned with a wider tau (e.g. val 40%-quantile → ~24 flagged on test) to see if the rescue rate scales linearly with the budget, or if hurts start to creep in like in the GSM8K-transferred large-budget run. Cheap (1 GPU retry).
2. **SVAMP-tuned multi-seed random control**: as before, 2–3 more random-complement seeds put variance bars on the random side and clarify whether the small +1.67pt is real or noise.

(1) is more informative about the mechanism (rescue scaling) and probably the right next move. (2) is the obvious robustness pass.

Outside the SVAMP cycle, the **T>0 retry pool on GSM8K** (Phase E-Retry-Pool) is still the natural next step on the diversity axis.
