# Differential Targetability Followup — Interpretation

**Date**: 2026-05-14
**Companion artifacts**: `reliability_differential_20260514.{md,json}`
**Script**: `eval/scripts/analyze_reliability_differential.py`
**Status**: **Reframes E4/E6**. The reliability score does not predict where `prob_vote` beats `exp_only`; it predicts overall sample difficulty. The published `+1.52pt` operational lift is mostly a consequence of a favorable test split combined with global `prob_vote` advantage, not a targeting effect.

## The motivating tension

E4 (retry simulation) and E6 (threshold policy) both report the same operational rule:

> `if logistic_broad_score < 0.5845 -> prob_vote else exp_only`
> on GSM8K test: `67.42% → 68.94%` (`+1.52pt`)

But `reliability_retry_20260514.md` also reports:

> standalone `prob_vote` on GSM8K test: `70.08%` (`+2.66pt`)

So **applying `prob_vote` to every test sample is `+1.14pt` better than the reliability-gated policy.** That asks the obvious question: is the score doing any targeting, or is the reported `+1.52pt` just a consequence of `prob_vote` happening to be better overall on this split?

## What we measured

For each sample we computed `delta = is_correct(prob_vote) - is_correct(exp_only) ∈ {-1, 0, +1}` and asked whether the reliability score correlates with `delta` (the method-switch utility), separately from how it correlates with `y_exp` (overall correctness).

Then for each split and each budget we compared three retry orderings:
- `score-gated`: swap the bottom-k by reliability score (current operational policy)
- `random-gated`: swap a random k (same budget, no signal)
- `oracle-gated`: sort by `-delta` (perfect targeting, ceiling)

If the score has real targetability, `score-gated` should sit clearly between `random-gated` and `oracle-gated`. If it lies near `random-gated`, the score is not adding information about *which* samples benefit from the swap.

## Findings

### 1. The score is a difficulty score, not a method-switch score

`logistic_broad_score` Pearson with the three labels:

| Split | Pearson(score, delta) | Pearson(score, y_exp) | Pearson(score, y_prob) |
|---|---:|---:|---:|
| gsm8k_val | −0.003 | +0.452 | +0.446 |
| gsm8k_test | −0.129 | +0.480 | +0.446 |
| svamp | −0.029 | +0.428 | +0.424 |

The score is **strongly** correlated with overall correctness for both methods (`y_exp` and `y_prob` correlations are nearly identical at ~0.45 on the same data). It is **essentially zero**-correlated with `delta`. This is the cleanest possible signature of "captures difficulty, not switching utility."

The val Pearson with `delta` is −0.003. We are not measuring some weak but real method-switch signal — we are measuring noise.

### 2. Val/Test distribution is asymmetric and explains most of E6's "lift"

| Split | n | exp_only | prob_vote | prob_better count | exp_better count |
|---|---:|---:|---:|---:|---:|
| gsm8k_val (fit data) | 1055 | 70.62% | 69.57% | 12 | **23** |
| gsm8k_test (eval data) | 264 | 67.42% | 70.08% | 7 | **0** |
| svamp | 300 | 86.33% | 86.67% | 2 | 1 |

- On **val**, `prob_vote` is **worse** than `exp_only` by 1.05pt; `exp_better` outnumbers `prob_better` almost 2:1. Any swap is on average a hurt.
- On **test**, `prob_vote` is strictly better — every `prob_better` is a free fix and `exp_better` is empty. Any swap is at worst a no-op.

So the policy was tuned on the side where swapping is generally bad, and then evaluated on the side where swapping has no downside. The reported "win" of `+1.52pt` is in large part a one-sided distribution artifact at n=264.

### 3. The targeting effect that does exist is small

`gsm8k_test` at budget 25%:

| Order | Fixes | Hurts | Net | Acc |
|---|---:|---:|---:|---:|
| score-gated | 4 | 0 | +4 | 68.94% |
| random-gated (expected) | 1.75 | 0 | +1.75 | ≈67.95% |
| random-gated (observed seed=42) | 1 | 0 | +1 | 67.80% |
| oracle-gated | 7 | 0 | +7 | 70.08% |
| **prob_vote global** | **7** | **0** | **+7** | **70.08%** |

Score-gated catches **4 of 7 fixes in the bottom 25%** (vs. random expectation 1.75). That is a real concentration — the score is doing something. But:
- `prob_vote` globally captures all 7 fixes at zero cost; the gating is strictly dominated.
- On the fit split (`gsm8k_val`) the same comparison gives score-gated net ≤ random-gated net at every budget ≥ 25%, and score-gated is actually negative at 40–50% budgets while random is closer to 0. The "targeting" effect is not robust to a budget choice.

### 4. SVAMP shows the same shape, smaller magnitude

`svamp` is right at the prob_better / exp_better noise floor (2 vs 1). Score-gated at 25% net `+1` vs random-gated `−1` — directional sign matches, but n=300 with 3 informative samples is not enough to call this a real targeting result either.

## What this means

- The published Phase E4/E6 operational rule is **not wrong**; it does produce a real test-time `+1.52pt`. But the framing "the reliability score targets samples that benefit from `prob_vote`" is **not supported** by the evidence. The mechanism is:
  - Global `prob_vote` advantage on the test split (lucky n=264 distribution): the dominant factor.
  - Weak score targeting (4/7 fixes in bottom 25% on test): a small contributor.
- On the fit split (val), and on SVAMP, the targeting effect is at or below random. The selection ability of the score for *method-switch utility* is fragile to which split you evaluate on.
- The score behaves exactly like what its fitting objective specified: it was trained against `y_exp` (correctness), so it learned "this sample is hard," not "this sample would benefit from a different voting method."

## What to change

1. **Update Phase E4/E6 narrative.** The score is a *difficulty* score with abstention-grade targeting (Phase E AURC results are clean). It is not a *method-switch* score, and the threshold policy should be reframed as "we observe `+1.52pt` on a small held-out split, but `prob_vote` globally gives `+2.66pt` on the same split, so the gating is not the load-bearing piece." The operational best rule on the current artifact is **use `prob_vote` for GSM8K test and SVAMP** — not the threshold gate.

2. **If we want a real method-switch score**, fit a classifier directly on `delta`, not `y_exp`. Concretely:
   - target = `1[is_correct(prob_vote)] − 1[is_correct(exp_only)]` (a {−1, 0, +1} signal)
   - features: the same per-sample feature set
   - protocol: same val/test split, plus multi-seed outer split for stability
   - acceptance bar: score-gated swap on val must beat random-gated swap at the chosen budget, not just match it.

3. **Multi-seed robustness on the GSM8K val/test split.** The `exp_better=0` count on the test split is a strong distributional accident. A 10-seed outer split would tell us how much of the `+1.52pt` is genuine and how much is one-shot evaluation noise. This is cheap (script support already exists) and should be done before promoting any rule to "operational."

4. **Drop the threshold rule from the "current operational best" position in the plan/tracker** until a multi-seed pass confirms it. Keep Phase E's abstention result (AURC, SelAcc@80%/@50%) as the load-bearing positive finding — that one is robust to the splitting question because it does not depend on which method is better.

## Caveats

- This pass uses only the one outer split (seed=42). A clean conclusion still wants the multi-seed pass mentioned above.
- The differential signal exists in absolute counts (7 prob_better on test, with score concentrating 4 of them in bottom 25%). With n=264 test and budget=25%, the targeting effect — if it is real — has roughly a 1–2 sample magnitude. Not zero, but not a foundation for an operational claim.
- The conclusion does not invalidate Phase E. The abstention result is a separate axis with a clean positive signal (combined AURC sits ~50% between random and oracle on val and test). The differential weakness is specifically about method-switching, not about reliability-as-difficulty.
