# Phase 8 Offline Proxy — Interpretation Notes

**Date**: 2026-05-13
**Companion artifacts**: `gsm8k_commit_gap_proxy_20260513.{md,json}`
**Script**: `eval/scripts/analyze_commit_gap.py`
**Status**: Phase 8 **deferred / unlikely to pay off** based on proxy.

## What the proxy does and doesn't measure

The true Phase 8 hook records the top1−top2 logit gap at the instant `select_indices` transfers a token (commit-time, per-token). The existing debug data does not have that — it only has window=5 logit gap at every step for the parsed-answer window. So the proxy redefines "commit step" using the `parsed_answer` trajectory:

| Proxy | Definition |
|---|---|
| `first_appear` | first step with non-null parsed_answer |
| `last_change` | last step where parsed_answer flips |
| `max_gap` | step with largest window-mean gap (`raw_weight`) |
| `mean_gap` | average of `raw_weight` across all valid events (≈ existing cgap_rawsum signal) |

This is a **step-level, window-mean** proxy — not per-token at commit. It tells us whether **one chosen step's gap** carries more correctness signal than the **diffuse temporal mean**.

## Headline numbers

### Single-event proxy vote (one event per sample)

| Method | Val (n=1055) | Test (n=264) | SVAMP (n=300) |
|---|---:|---:|---:|
| exp_only (baseline) | **70.62%** | 67.42% | **86.33%** |
| proxy_first_appear | 34.41% | 34.47% | 72.33% |
| proxy_last_change | 69.95% | 67.05% | 86.00% |
| proxy_max_gap | 69.86% | **67.80%** | 86.00% |

### Per-sample correlation with baseline correctness (full GSM8K, n=1319)

| Feature | Pearson | Spearman |
|---|---:|---:|
| **max_gap** | **+0.332** | **+0.332** |
| last_change_gap | −0.269 | −0.239 |
| mean_gap | +0.100 | +0.115 |
| first_appear_gap | −0.007 | +0.023 |
| last_change_step_frac | −0.308 | −0.304 |
| n_unique_answers | −0.277 | −0.303 |
| n_valid | +0.183 | +0.164 |
| max_gap_step_frac | +0.124 | +0.082 |
| first_appear_step_frac | −0.141 | −0.170 |

## Findings

### 1. One commit-equivalent step ≈ entire temporal vote

`last_change` and `max_gap` proxies both reach ~99% of exp_only val/test/SVAMP accuracy using a **single event per sample**. Specifically:
- val 69.86 vs 70.62 (gap 0.76pt = 8 samples)
- test 67.80 vs 67.42 (max_gap +1 sample over baseline)
- SVAMP 86.00 vs 86.33 (gap 1 sample)

**Implication**: the exp temporal voting machinery is mostly redundant — the eventual answer is essentially **decided at one settlement step** and exp_only just confirms it with mass. This explains why Phase 7/7B/7-filter could not beat baseline: there's no room — the baseline is already near a single-event ceiling.

### 2. max_gap correlation (+0.332) is 3× stronger than mean_gap (+0.100)

This is the first genuine new finding from Phase 8 work: **peak gap is much more predictive of sample-level correctness than the diffuse temporal mean**. The signal exists at the sample level but it is **between-sample**, not within-sample.

| Signal axis | Source | Strength |
|---|---|---|
| Within-sample ranking | Phase 3 win-rate (correct vs wrong events in same sample) | 0.86 (strong) |
| Between-sample reliability | Phase 8 proxy max_gap pearson (this work) | +0.33 |
| Aggregated weight signal (sum) | Phase 7 hybrid, Phase 7-filter | weak (noise) |

These three are **different uses of the same gap data**, and the strong signals (within-sample ranking, between-sample reliability) **do not aggregate well into the form that voting consumes** (per-event weight summed across events).

### 3. last_change_step_frac (−0.308) and n_unique_answers (−0.277)

Two structural indicators of sample-level difficulty:
- Samples whose answer settles late (high step_frac) tend to be wrong.
- Samples that produce many distinct candidate answers tend to be wrong.

These are stronger than any gap feature except max_gap.

### 4. Negative last_change_gap correlation (−0.269)

When the model's final-stage answer flip has a **high** window gap, the sample is more likely **wrong**. Interpretation: a high-confidence late flip indicates the model still had a competitor at the late step — fragile state. Combined with the negative `last_change_step_frac` correlation, this paints the same picture: late + decisive flips = unstable sample.

## What this means for the true Phase 8 GPU hook

The hook would give true per-token commit gap (instead of window-mean step gap). A reasonable a-priori expectation is that commit-gap is a *sharper* version of step-gap, since it captures the gap exactly at the moment the model fixes the token. But the proxy results say:

1. Step-level gap aggregated at any single proxy step **already saturates exp_only** (no headroom).
2. The strongest per-sample correlation we found (max_gap +0.332) **does not translate into vote accuracy gain**.
3. The voting routine itself is the bottleneck — even an oracle within-sample ranker (Phase 3 0.86) failed to lift accuracy (Phase 7-filter NEGATIVE).

A true commit-gap signal would need to **outperform** these on **vote aggregation**, not just on correlation. There is no proxy result suggesting that's likely. Specifically:
- If commit-gap is just sharper than step-gap, the sample-level correlation might rise (e.g., +0.40), but voting integration would still be capped at single-event-proxy ≈ baseline.
- For commit-gap to **lift vote accuracy** above baseline, it would need to differentiate **which of multiple events in a sample** has the *correct* answer — but Phase 3's 0.86 win-rate already showed that signal exists and Phase 7 family showed it's not vote-aggregable.

## Verdict

- **Hybrid line definitively closed.** Phase 7/7B (sum), Phase 7-filter (prune), Phase 8 proxy (single-event commit) all fail to beat exp_only on test. Three independent activation paths for the gap signal in voting all return null.
- **GPU rerun for true commit-gap hook: not recommended.** The proxy upper-bounds the value of commit-gap in voting. Spending GPU on the real hook would test the same hypothesis with more fidelity but the prior is strongly negative.
- **New positive finding worth preserving**: `max_gap` pearson +0.332 (sample-level reliability), `n_unique_answers` −0.277, `last_change_step_frac` −0.308. These are **sample-level confidence signals**, not vote-weighting signals. Could be useful for:
  - Selective abstention / asking model to retry
  - Confidence calibration (well outside the dLLM-MidTruth scope)
  - Per-sample weighting in **ensembles** (between samples, not within)

## Recommended next direction

- Close the hybrid line in plan/tracker with the Phase 8 proxy as the final entry.
- Pivot to one of:
  - **(B)** Math500 Bucket A parser-side analysis (user is taking this).
  - **(E)** Use max_gap / n_unique_answers as an **abstention or retry filter** rather than vote weight — different application axis where the +0.33 sample-level signal is the right shape.
  - Or step away from the gap-signal family entirely and revisit when generation-quality work surfaces a new structural lever.
