# E-Diff2 Multi-seed Pass — Interpretation

**Date**: 2026-05-14
**Companion artifacts**: `reliability_multiseed_20260514.{md,json}`
**Script**: `eval/scripts/analyze_reliability_multiseed.py`
**Status**: **Partially reverses E-Diff's single-seed conclusion.** Across 10 outer seeds, the E6 threshold-gated swap is a small but consistent improvement over both `exp_only` and global `prob_vote`. The seed=42 result that drove both E4/E6 (`+1.52pt`) and the E-Diff counter-claim (`prob_vote dominates`) was inflated relative to the multi-seed mean.

## The setup

The same logistic_broad pipeline is refit on val per seed (`seed ∈ [42, 51]`, val_frac=0.8). The threshold `tau` is chosen at the val 25%-quantile of the reliability score (matches E6). Test reports `exp_only`, global `prob_vote`, and gated acc.

## What the numbers say

| Strategy | Mean acc | Std | Beats exp_only | Beats gated |
|---|---:|---:|---:|---:|
| exp_only | 69.05% | 1.96pt | — | 30% |
| prob_vote global | 68.94% | 1.93pt | 40% | 20% |
| **gated** | **69.66%** | **1.82pt** | **70%** | — |

Per-seed deltas:
- gated − exp_only: **+0.61pt ± 0.78pt**
- prob_vote − exp_only: **−0.11pt ± 1.55pt**
- gated − prob_vote: **+0.72pt ± 1.11pt**

prob_better / exp_better on test, across seeds: `mean ± std = 4.60 ± 1.43 / 4.90 ± 3.18`. Seed=42's `exp_better=0` is the extreme low end; seed=49's `exp_better=11` is the extreme high end.

## What changed vs E-Diff's single-seed conclusion

E-Diff (seed=42 only) read: "global `prob_vote` beats gated by `+1.14pt` on test; the gating is dominated; the `+1.52pt` lift is a distributional artifact."

Multi-seed reads: "global `prob_vote` is essentially neutral vs `exp_only` (mean `−0.11pt`); the gating is positive and stable (mean `+0.61pt`); gating > global swap on 70% of seeds."

The E-Diff framing was right that seed=42 had a favorable distribution for `prob_vote` (test `exp_better=0`). It was wrong to conclude that this implied `prob_vote` is globally better — on most other seeds, the test split has nonzero `exp_better` counts, and global swap pays for those.

## Why the gating still helps when `Pearson(score, delta) ≈ 0`

The differential analysis showed `Pearson(logistic_broad_score, delta) ≈ −0.003` on val and `−0.129` on test. The score does not strongly predict where `prob_vote` will help vs hurt.

But the multi-seed result still shows gating helps on average. The reconciliation is in the **budget constraint**:

- Global `prob_vote` is a 100% budget swap. It captures all `prob_better` cases but also pays for all `exp_better` cases. With `exp_better` mean 4.90 vs `prob_better` mean 4.60, the global swap is slightly negative-expected on average.
- The gated rule is a 25%-budget swap, targeted at the lowest-reliability samples. Even a weak (small Pearson) targeting signal, combined with limiting the swap to 1/4 of the samples, is enough to skew the fix:hurt ratio favorably.

The mechanism is **budget-bounded swap + weak targeting**, not strong targeting. Looking at per-seed `fixes/hurts`:
- seed 47: 4/0 (clean win)
- seed 42: 4/0 (clean win)
- seed 51: 3/0 (clean win)
- seed 49: 1/3 (loss, but bounded to budget 25%)
- seed 45: 1/2 (loss, also bounded)

The gating doesn't always win, but when it loses it loses small (bounded by the budget), and when it wins it can win clean.

## Where the load-bearing finding actually is

- **Phase E abstention** remains the load-bearing positive: combined AURC `0.2515` on gsm8k_test sits 50% of the way from random to oracle, and the SVAMP transfer is the same shape. That result does not depend on which voting method is the baseline.
- **The E6 threshold-rule swap** has a real but small mean lift over `exp_only` (`+0.61pt ± 0.78pt`). It also has a real but small mean lift over global `prob_vote` (`+0.72pt ± 1.11pt`). These are not noise — 70% of seeds favor gated over both alternatives — but the per-seed magnitude is much smaller than the single-seed `+1.52pt` originally reported.
- **The E-Diff differential finding** is correct on the literal point (`Pearson(score, delta) ≈ 0`), but the operational implication ("therefore use prob_vote globally") was wrong. The right operational implication is "use a low-budget gated swap, because budget + weak targeting > unconstrained swap on average."

## Recommended plan/tracker updates

1. Move the operational rule (`logistic_broad_score < tau → prob_vote`) back into the "current best on artifact" position, with the corrected magnitude `+0.61pt ± 0.78pt` (not `+1.52pt`).
2. Keep the differential finding intact: `Pearson(score, delta) ≈ 0` is real and informative — the gating works through budget bounding, not through targeting strength.
3. Drop the "use prob_vote globally" recommendation that E-Diff's single-seed read implied. Global swap is high variance (`±1.55pt`) and not better than exp_only on average.
4. The E-Diff3 idea (re-fit a classifier directly on `delta`) is still worth trying, but with the multi-seed result in mind: even with `Pearson ≈ 0` targeting, gating works. A delta-trained score might give a bigger lift, but the gain over the difficulty-trained score is likely modest given that budget bounding is doing most of the work.

## Caveats

- 10 seeds is small. The `+0.61pt ± 0.78pt` 95% CI is roughly `[+0.13pt, +1.09pt]` — barely above zero. A 30-seed pass would tighten this and is cheap.
- All 10 splits draw from the same 1319-sample artifact. Seeds differ only in which 264 go to test. A more aggressive robustness check would use a different inference run (e.g., the `_prob_mean_rawsum_bs4_all_debug` artifact), but that requires re-extracting features and is outside this pass's scope.
- The val/test mismatch on the `prob_vote vs exp_only` question (val favors exp, test sometimes favors prob) is itself a real signal about the underlying voting method comparison. A separate "which voting method should we deploy" question is not what this script answers and is not what the gated rule is for.
