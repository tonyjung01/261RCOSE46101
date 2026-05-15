# Phase E-Retry-Control-MultiSeed — Interpretation

**Date**: 2026-05-15
**Companion artifacts**: `retry_control_multiseed_eval_20260515.{md,json}`
**Retry artifacts**: `outputs/.../20260514_gsm8k_retry_exp_complement_seed{124,125,126,127}/`
**Status**: **Targeting marginal robust under random-seed variation on this outer split.** Single-outer-split, single retry per seed, but the random-control side is now multi-seeded with `K=4` complement controls.

## Setup

The selective retry result (`P_selective = 70.08%`, `+2.65pt`) was a single point; both the original pure-random control and the complement-only control gave single-seed point estimates for the targeting marginal. This pass runs the complement-only control with 4 random seeds (`124, 125, 126, 127`, all `--exclude-flagged` so `0/60` overlap with the score-flagged set) to put a variance estimate on that marginal.

Selective retry, outer split, score, threshold, and answer kind are all held fixed; only the random-control draw varies.

## Headline numbers

### Per-seed results

| seed | overlap | P_random | Δ vs base | random subset (base → retry, Δ) | targeting marginal (`P_selective − P_random`) |
|---:|---:|---:|---:|---|---:|
| 124 | 0/60 | 67.05% | `−0.38pt` | 76.67% → 75.00% (`−1.67pt`) | **`+3.03pt`** |
| 125 | 0/60 | 66.29% | `−1.14pt` | 80.00% → 75.00% (`−5.00pt`) | **`+3.79pt`** |
| 126 | 0/60 | 67.80% | `+0.38pt` | 78.33% → 80.00% (`+1.67pt`) | **`+2.27pt`** |
| 127 | 0/60 | 67.05% | `−0.38pt` | 85.00% → 83.33% (`−1.67pt`) | **`+3.03pt`** |

### Aggregate (K=4 random-control seeds, complement-only)

| Metric | Mean ± std |
|---|---|
| targeting marginal `(P_selective − P_random)` | `+3.03pt ± 0.62pt` |
| approximate 95% CI on the mean | ≈ `[+2.42pt, +3.64pt]` |
| range across seeds | `[+2.27pt, +3.79pt]` |
| random subset-local retry delta | `−1.67pt ± 2.72pt` |
| seeds with positive targeting marginal | `4 / 4` |
| seeds with negative random-subset retry delta | `3 / 4` |

## Findings

### 1. Targeting marginal is robust to random-control seed

All four random-control seeds give a positive targeting marginal. The mean is `+3.03pt` with std `0.62pt`. A `95%` CI on the mean (using `t ≈ 3.18` for K=4) is roughly `±0.99pt`, so `[+2.04pt, +4.02pt]` — comfortably above 0. With the simpler `1.96 · std/√K` approximation it tightens to `[+2.42pt, +3.64pt]`. Either way, the targeting marginal is not a single-seed accident on this outer split.

The earlier pure-random control's `+2.27pt` (seed=123, `12/60` overlap) sits at the low end of this 4-seed range. That is consistent with the hypothesis that the 12 flagged samples leaking into the pure-random control were slightly deflating the targeting marginal; the complement-only multi-seed estimate is its cleaner version.

### 2. "Retry on unflagged is net-negative" is the directional pattern, not a strict claim

3 of 4 seeds show a negative random-subset retry delta (`−1.67pt`, `−5.00pt`, `−1.67pt`), but seed=126 shows `+1.67pt`. Mean `−1.67pt` with std `2.72pt` is wide and not significantly below 0. So:

- **Defensible claim**: on this artifact, retry on a random unflagged subset is *on average* net-negative or near zero, in clear contrast to the `+11.67pt` retry delta on the score-flagged subset.
- **Not yet defensible**: that retry on unflagged is strictly negative on every random draw.

The asymmetry between `selective subset delta = +11.67pt` and `random complement subset delta mean = −1.67pt ± 2.72pt` is the right level of summary: even the worst (most favorable) random complement subset (`seed=126`, `+1.67pt`) is dramatically smaller than the selective subset's `+11.67pt`.

### 3. Range is narrow

Targeting marginal range across 4 seeds: `[+2.27pt, +3.79pt]`, span `1.52pt`. On `n=264` test that is `~4 samples` of spread between the worst and best random draws. The seed-to-seed wobble is small relative to the central tendency `+3.03pt`. This is consistent with "the random control distribution is reasonably tight here" — not surprising given fixed selective and fixed retry pipeline.

## What this strengthens

- **The targeting claim** is no longer just "single-seed `+2.27pt`". It is `+3.03pt ± 0.62pt across 4 random-control seeds, all positive`. This survives the most plausible single-seed-noise objection.
- **The decomposition story** stays consistent: the score concentrates retry budget on a slice (flagged 60, base `30%`) where retry has positive expected value; on any random draw from the complement (base `~78%`), retry has near-zero or negative expected value. Multi-seed makes this asymmetry harder to dismiss.

## What this still does NOT establish

- **Outer-split robustness**: the score, selective subset, and retry artifact are all on `seed=42` outer split. A different outer split would re-select a different flagged 60 and re-fit the score; we have not measured that. The `+3.03pt ± 0.62pt` is the random-control variance with selective held fixed.
- **Cross-task transfer**: GSM8K only. SVAMP and Math500 retry effects unknown.
- **Cost-efficiency multiples**: still no fixed ratio (e.g. "~Nx more efficient"); the right level remains "+11.67pt on flagged vs ~−1.67pt on complement, single outer split, K=4 random seeds."
- **Strict "net-negative on unflagged"**: 3/4 seeds support it, 1/4 contradicts, std is wide; treat as a directional finding, not a constant.

## Recommended next step

The targeting claim is now in reasonably good shape at single-outer-split. The two remaining axes that would meaningfully strengthen the picture, in priority order:

1. **Multi outer-split (Phase E-Retry-OuterSeed)** — repeat the entire pipeline (score fit on val, flag bottom 22.73% of test, retry, evaluate) for `2–3` additional outer splits. This is the only direct test of whether the selective-retry story survives changing which 264 test samples we draw. Cost: per outer split = `1` selective retry GPU run + `≥1` random control GPU run + score-fit (offline).
2. **Phase E-Retry-Pool (real K-pool, T>0)** — orthogonal axis. With targeting confirmed at single outer split, the diversity question (does aggregating K independent regenerations on the same flagged 60 beat K=1?) becomes the natural next refinement.
3. **Cross-task confirmation** — does the same pipeline yield similar targeting marginal on SVAMP (currently 86.33% base)? Cost: selective retry on SVAMP flagged subset + random-control SVAMP retry.

Given the cost asymmetry, (1) and (3) are the higher-information moves; (2) is more about understanding mechanism than confirming the claim.
