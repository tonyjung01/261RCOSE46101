# True Retry Decomposition — Interpretation

**Date**: 2026-05-14
**Companion artifacts**: `true_retry_decomposition_20260514.{md,json}`
**Script**: `eval/scripts/analyze_true_retry_decomposition.py`
**Status**: **Clean positive structural finding.** Decomposes the published `+2.65pt` true-retry result into "offline-capturable fixes" and "regeneration-diversity fixes", and shows the latter is non-trivial.

## What we set out to ask

The published P2 result is `+2.65pt` on GSM8K test (8 fixes, 1 hurt), versus the offline P1 fallback's `+1.52pt` (4 fixes, 0 hurts). Both policies act on the same 60 flagged samples (score below tau=0.5845, 22.73% of test). The natural question: **is true retry just "offline fallback plus noise", or is it capturing genuinely new fixes that no static fallback in the existing artifacts could produce?**

## Findings

### 1. On the current artifact, retry fixes contain the offline fallback fixes

| Source | Fix set on flagged 60 (seed=42, this artifact) |
|---|---|
| offline fallback (P1) | `{124, 200, 411, 471}` |
| true retry — exp on retry (P2) | `{124, 200, 411, 451, 471, 538, 646, 1093}` |

On the current seed=42 artifact, the retry fix set strictly contains the offline fallback fix set. The 4 retry-only samples (`451, 538, 646, 1093`) are samples where the offline `prob_vote` answer was wrong in *this* artifact, so the two available static fallback sources (`exp_only`, `prob_vote`) do not rescue them.

This is an artifact-level observation, not a structural claim. A richer offline pool (e.g. additional seeds, more voting variants, blockactive) could shrink the retry-only set; a different outer-split seed could shift which samples land in the flagged 60 in the first place.

### 2. Roughly half of retry fixes are not recoverable from the available offline fallback sources

"Uniquely retry-rescued" = (base wrong) AND (prob wrong) AND (retry right) → `4/60` on this artifact. These are samples where neither of the two static answers available in the offline pool was right, but the regeneration produced the correct answer.

Arithmetically: 4 unique-to-retry fixes + 4 shared fixes = 8 total. So on this artifact, **roughly half of the retry fixes are not recoverable from the available offline fallback sources**. We avoid the stronger phrasing "half comes from regeneration variance" because that asserts a stable mechanism share — and we have not yet shown that the 4 unique fixes wouldn't shrink under a richer offline pool or a different seed.

### 3. Vote method on the retry run does not change the answer on this artifact

| Source | Correct on flagged 60 |
|---|---:|
| retry_exp answer | 25 |
| retry_exp vote answer | 25 |
| retry_prob answer | 25 |
| retry_prob vote answer | 25 |

All four candidates land at exactly 25/60 on the current artifact. Reading `exp_only`, `vote_answer` from an exp run, or `vote_answer` from a prob run produces the same set of correct samples. On this artifact, the gain therefore does not depend on which voting method is read off the retry. Whether this generalizes — i.e. whether vote method choice is broadly irrelevant once regeneration is in play — is a hypothesis consistent with this single artifact but not established by it.

### 4. The score targets samples with real retry headroom

Base accuracy stratified by flagged status (on test n=264, current artifact):
- flagged 60 base acc: `30.00%`
- unflagged 204 base acc: `(178 − 18)/204 = 78.43%`
- overall test base acc: `67.42%`

`logistic_broad_score` is doing real difficulty targeting on this artifact: it isolates a slice of test that is ~`2.6×` worse than the population mean. Per-sample retry net improvement rate on the flagged slice = `7 net / 60 ≈ 11.7pt per retried sample`. On the unflagged slice the per-sample improvement is bounded by the gap to 100% (i.e., at most `21.6pt` headroom on average and likely a small fraction of that).

This pattern is **consistent with** selective retry being more cost-effective than uniform retry, but we have not measured the unflagged retry rate directly. We avoid quoting a fixed efficiency multiple (e.g. "~6×") until E-Retry-Control directly compares score-targeted retry against a random-budget retry of the same size.

### 5. The retry hurt is the regeneration-variance cost on this seed

Sample `727` is the single retry hurt: base was correct, retry produced a different (wrong) answer. There is no static-fallback equivalent in the available offline pool — `prob_vote` happens to agree with base on this sample.

Hurt rate on the flagged 60 = `1/60 ≈ 1.7%`, on this seed. With 4 unique fixes and 1 hurt, the retry-over-offline net is `+3` here. The hurt rate sets a single-seed reference for the regeneration variance cost; multi-seed retry would tighten it.

## What this implies for the operational story (working hypothesis, not closed)

A consistent reading of all current results is:
- **Reliability score** does real difficulty targeting (flagged 30% vs unflagged 78% on this artifact). It is not a method-switch predictor (Pearson(score, delta) ≈ 0), but it does not need to be in this framing — it just needs to spot low-confidence samples that are worth regenerating.
- **Selective regeneration** then captures two observable types of fix on this artifact:
  - **shared fixes** (4 here): samples where the existing `prob_vote` answer was already right; offline fallback also catches these
  - **retry-only fixes** (4 here): samples where neither static fallback in the available offline pool was right, but a fresh trajectory was; offline policies built only from `exp_only`/`prob_vote` cannot catch these
- The "retry beats offline fallback by `+1.13pt`" gap on this artifact is consistent with the retry-only slice (`+4`) and a single-sample hurt (`−1`).

The mechanism — "difficulty-targeted regeneration that picks up new correct answers no offline source had" — is plausible. But the magnitudes are single-seed, single-artifact, and the offline pool here is small (only two static sources). Stronger framings (constant efficiency multipliers, exact share of gain attributable to regeneration variance) need direct controls before being asserted.

## Caveats

- One outer seed (=42), one retry run for each of exp/prob vote methods. The multi-seed pass (`reliability_multiseed_20260514`) covered only the offline policies; retry was not multi-seed because each seed would require a fresh GPU rerun. The decomposition above is structural enough to be informative on one seed, but the magnitudes (4 unique fixes, 1 hurt) are stochastic.
- The diversity count (4) is sensitive to which `prob_vote` artifact is used as the "existing offline source." A richer offline pool (multiple seeds, multiple voting methods) would shrink the diversity slice somewhat, because more original artifacts means more chances of one of them being right.
- The 60 flagged samples are sampled at the seed=42 val score quantile. Different val splits would flag slightly different sets, and the per-flagged-sample retry success rate (11.7pt) might shift.

## Recommended followups

1. **Cost-efficiency framing in the tracker.** The retry result is not just an accuracy gain; it is a selective-compute story (`22.73%` retry budget → `+2.65pt` test lift, matching what global retry would deliver if it followed the same per-sample uplift on the high-confidence slice — but that bet would be very expensive and likely capture little additional fix headroom).
2. **Random-budget retry control.** Spend the same `60` retry slots but on a random subset of test. If random-budget retry yields, say, `+1.0pt`, then selective adds `+1.65pt` of pure targeting value. This needs one extra GPU run with the same total cost (`60` regenerations) and is the cleanest measurement of how much of the `+2.65pt` is "right samples to retry" vs "retry helps in general."
3. **Diversity-pool retry.** If we have N independent regeneration seeds, "uniquely rescued" should grow approximately as `1 − (1 − p_rescue)^N`. Even a small N=2 retry pool could lift the diversity slice meaningfully. This is the natural extension of the regeneration-variance lever.
4. **Multi-seed retry decomposition (expensive).** Re-flag and re-run for several outer seeds. With the current single-seed structural result already showing diversity-fixes ≈ method-switch-fixes, multi-seed would mostly tighten the magnitudes rather than change the qualitative picture.
