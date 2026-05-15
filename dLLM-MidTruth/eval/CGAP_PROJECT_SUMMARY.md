# cgap Project Summary

**Date**: 2026-05-15
**Scope**: confidence-gap voting and reliability-triggered selective retry for `dLLM-MidTruth`
**Companion docs**: detailed phase records in `CONFIDENCE_GAP_EXPERIMENTS.md`, implementation plan in `CGAP_IMPROVEMENT_PLAN.md`, per-pass interpretations in `eval/analysis/*_interpretation.md`

This document is a consolidated, honest summary of what the cgap project established, what it ruled out, and what should be treated as a separate technique. It is intentionally claim-graded — each result is tagged with the strength of evidence we currently have.

---

## 1. Project arc — original goal and what shifted

**Original goal**: evaluate whether confidence-gap signals can replace or complement the time-based exp voting weights from dLLM-MidTruth's TSCV. The hope was that gap-derived weights produce a stronger vote than the exp temporal weight.

**What actually happened**: the gap signal is real (it correlates with sample-level correctness), but **not vote-aggregable** in the original sum-over-events form. The project pivoted from "gap as vote weight" to "reliability score for sample-level compute allocation", and the second framing produced a clean small positive operational result.

The full trajectory:

1. **Vote-weight track (Phase 7-family) — failed**
2. **Reliability / abstention track (Phase E and follow-ups) — succeeded on GSM8K, weakly succeeded on SVAMP**
3. **Self-consistency / K-pool probe — separate technique, weak gain, methodology-distinct**

Sections 2–4 below cover each layer.

---

## 2. Layer 1 — gap as vote weight: **ruled out**

The hypothesis: build a per-step gap-weighted vote that beats `exp_only` temporal voting.

| Pass | Mechanism | Result |
|---|---|---|
| Phase 7 (sum hybrid) | `exp_weight + λ·quality(gap)` additive / multiplicative | test 67.42% (= exp_only baseline); no lift |
| Phase 7B (exp-scale relaxation) | `max_norm` / `step_frac` exp normalization, wider λ | val `71%` but test still `67.42%`; v0 result was noise |
| Phase 7-filter (gap pruning) | drop low-gap events from vote pool | best val q=0 (no filtering); test `67.42%` |
| Phase 8 (commit-time gap offline proxy) | single-event commit-equivalent voting | `proxy_last_change` / `proxy_max_gap` reach `~99%` of exp_only — voting saturates at one settlement step |

**Phase 8 was the decisive negative**: a single proxy event recovers exp_only accuracy, meaning the voting itself has very limited headroom on this artifact. There is no room for gap reweighting to help.

**Conclusion**: on the current artifact, the gap signal is not vote-aggregable. The hybrid line is closed.

This is a clean negative — four independent activation paths (sum, scale relaxation, prune, commit-time proxy) all fail to lift test accuracy above `exp_only`.

---

## 3. Layer 2 — reliability score for selective compute: **the load-bearing positive**

**Pivot**: the gap signal is per-sample reliability information, even if it does not aggregate inside a vote. Use it to decide *where to spend additional compute*, not to weight votes.

### 3.1 Phase E — sample-level abstention works (first positive)

Per-sample reliability features (`max_gap`, `n_unique_answers`, `last_change_step_frac`, etc.) drive a real abstention signal.

| Strategy | GSM8K test AURC | random | oracle |
|---|---:|---:|---:|
| zsum_combined | 0.2515 | 0.3034 | 0.1521 |
| logistic_broad | **0.2354** | 0.3034 | 0.1521 |

Combined score sits about **half-way between random and oracle** AURC. SelAcc@80% gain on test: `74.88% → 77.73%` (logistic_broad). Cleanly above random. **Strong evidence**.

### 3.2 Selective retry on GSM8K — single-seed positive

`logistic_broad_score < 0.5845` flags 60/264 (`22.73%`) of GSM8K test. Re-running just those samples and keeping `exp_only` elsewhere:

| Metric | Value |
|---|---|
| baseline `exp_only` | 67.42% |
| **selective retry (T=0, v6)** | **70.08% (`+2.65pt`)** |
| fixes / hurts on flagged | 8 / 1 |
| flagged subset base acc | 30.00% (vs unflagged 78.43%) |

**Score is concentrating retry budget on samples with real retry headroom**. Single seed, single retry; magnitudes artifact-level. **Strong direction, moderate magnitude**.

### 3.3 Random-budget control — targeting is real

To isolate "selective targeting" from "retry helps in general", we ran budget-matched random retries (same 60-sample budget). After tightening to a complement-only random control (overlap with flagged = 0) and aggregating across `K=4` random-control seeds:

| Quantity | Value |
|---|---|
| targeting marginal `(P_selective − P_random)` mean | `+3.03pt ± 0.62pt` |
| 4/4 seeds positive | ✓ |
| approx 95% CI on mean | `[+2.42pt, +3.64pt]`, cleanly excludes 0 |
| random subset-local retry delta | `−1.67pt ± 2.72pt` (3/4 negative) |

**Scope reminder**: this is `K=4` random-control seeds with the outer val/test split fixed at seed=42. It establishes that the targeting is not a single-random-draw artifact on this outer split. It does **not** establish robustness across new val/test splits — that experiment is deferred. **Strong on this outer split, deferred at higher level**.

### 3.4 GSM8K T=0 budget sweep — monotone growth

For `k ∈ {5, 10, ..., 60}` of bottom-by-score flagged samples, apply retry only on those. Fixes grow `1 → 2 → 2 → 3 → 4 → 4 → 5 → 6 → 8`; hurts flat at `1`; fix/hurt ratio improves `2.0 → 8.0`. No diminishing-returns signal up to k=60.

**Conclusion**: `tau=0.5845` (k=60) is not over-budgeted; the current GSM8K artifact is consistent with a wide retry-useful budget across the score-sorted flagged population. We cannot probe `k > 60` without new GPU runs. **Clean shape; saturation not observed**.

### 3.5 SVAMP cross-task — mixed transfer; budget shape is different

Applying the GSM8K-fitted score unchanged to SVAMP: `+0.00pt` selective retry lift (3 fixes balanced by 3 hurts). Refitting the score on SVAMP val: `+1.67pt` selective lift with 1 fix / 0 hurts at the tight budget (q25). Widening tau to q40 added `+1` fix and `+1` hurt — same net effect, more risk. **Weak positive on SVAMP-tuned, with diminishing returns at wider budget**.

Cross-task structural finding: same score, **opposite budget-scaling behavior** on GSM8K (monotone growth) vs SVAMP (diminishing returns). Driven by task base-acc distribution (GSM8K's unflagged is at 78%, SVAMP's at 93–100%; SVAMP unflagged is saturated so widening tau pulls in confident-borderline samples that retry can flip).

### 3.6 Per-base-wrong retry rescue rate is task-portable

| Setting | base-wrong in flagged | retry rescues | rate |
|---|---:|---:|---:|
| GSM8K (v6 selective) | 42 | 8 | `~19%` |
| SVAMP-tuned q25 | 6 | 1 | `~17%` |
| SVAMP-transferred (q ≈ tau=0.5845 on full SVAMP) | 22 | 3 | `~14%` (with 3 hurts) |

The per-base-wrong rescue rate sits in a narrow `14–19%` band across tasks. **Moderate-strength portable finding**, single-seed.

### 3.7 What this section establishes — and what it does not

**Established (artifact-level, single outer split)**:
- The reliability score is a real difficulty signal.
- Score-targeted selective retry on GSM8K gives a small positive lift (`+2.65pt`, `+3.03pt ± 0.62pt` targeting marginal vs complement random).
- The gain is not driven by random-subset noise (K=4 random controls, all positive).
- The lift mechanism partly transfers to SVAMP under a SVAMP-tuned score (`+1.67pt`).
- Per-base-wrong retry rescue rate is roughly portable (`~14–19%`).

**Not established**:
- Robustness across new val/test splits (outer-seed multi-seed deferred).
- Cross-task generality beyond SVAMP (Math500 not tested).
- That the score predicts *method-switch utility* (`Pearson(score, delta) ≈ 0` — see Phase E-Diff). The score is a difficulty signal; gating helps via *budget bounding* on top of weak targeting, not via switch-targeting.

---

## 4. Layer 3 — K-pool / self-consistency probe: **separate technique, weak gain, kept apart**

We also probed K-pool aggregation: take the same flagged 60, run `K=3` independent regenerations at `T=0.2` (with seed plumbing), aggregate.

**Why this is its own thing, not a continuation of Section 3**:

- Requires `T > 0` (deterministic at T=0); breaks the canonical T=0 evaluation framework.
- The aggregation step is mechanistically self-consistency / majority@K, not the reliability score's targeting effect. The reliability score's role here is the same as in Section 3 (selecting *which* 60 samples to retry); the K-pool gain on top is from regeneration diversity, not from the score.
- Operationally costs K× more compute on flagged samples.

**Results**:

| Method on flagged subset (n=60) | Correct |
|---|---:|
| base (no retry) | 18/60 (30%) |
| K=1 (T=0.2, seed=42) | 24/60 (40%) |
| K=3 majority | 26/60 (43%) — **+2 samples over K=1** |
| K=3 best-by-confidence (margin or top1) | 21/60 (35%) — **worse than K=1** |
| K=3 confidence-weighted vote | 22/60 (37%) — worse than K=1 |
| K=3 "2-of-K else base" | 25/60 (42%) — slightly worse than majority, cleaner fix/hurt ratio |
| K=3 union-any (oracle ceiling) | 35/60 (58%) — `+11` over K=1 |

Full-test deploy with majority(K=3): `70.45%` (`+3.03pt` vs base), only `+0.37pt` over K=1.

**Two mechanism findings**:

1. **Confidence aggregators (best-by-margin/top1, confidence-weighted) actively underperform K=1**. The within-retry vote margin does **not** predict which retry is correct. All 9 oracle-vs-majority gap samples are `1-1-1` splits where the "correct" retry's confidence is not systematically higher.
2. **"2-of-K else base" is the cleanest deployable policy** (`9 fixes / 2 hurts`, fix/hurt ratio 4.5) but its net lift is smaller than plain majority.

**Conclusion**: regeneration diversity exists on this artifact (oracle ceiling has `+11 sample` headroom over K=1), but it is **not exploitable with naive in-vote signals**. The deployable K-pool gain on top of K=1 is small (`+0.37pt`). The selective-retry-plus-K-pool deployment runs at K× compute for ~10% of the oracle headroom captured.

We do not consider K-pool a contribution of the reliability-score line; it is a separate self-consistency probe with mixed signals.

---

## 5. Cross-task structural finding (worth keeping)

The same reliability score (`logistic_broad`, fit on the respective task's val) shows two opposite budget-scaling behaviors:

| Task | base acc | flagged base | unflagged base | budget scaling |
|---|---:|---:|---:|---|
| GSM8K test (n=264) | 67.42% | 30.00% | 78.43% | **monotone growth** (fixes grow, hurts flat) |
| SVAMP test (n=60) | 86.67% | 57.14% | 95.65–100% | **diminishing returns** (fixes and hurts grow together past q25) |

**Reading**: the score's behavior depends on *task base-accuracy distribution*. On GSM8K, the score-sorted flagged set appears to have a wide retry-useful budget. On SVAMP, the unflagged complement is saturated; widening tau pulls in borderline-confident samples that regeneration can flip. **Same score quality, different task structure → different operational budgets**.

This is one of the clearest cross-task patterns in the project: it suggests that the selective-retry policy will need task-specific budget tuning when applied to a new dataset.

---

## 6. Operational recommendations (artifact-level, single outer split)

If the project's findings were deployed today on GSM8K with the current artifact:

| Goal | Policy |
|---|---|
| Best within canonical T=0 framework | `if logistic_broad_score < 0.5845 → retry that sample at T=0; keep retry's exp_only answer` (`+2.65pt` over baseline) |
| Absolute max among tested policies (separate T>0 self-consistency probe) | `K=3 majority` on the flagged subset at `T=0.2` (`+3.03pt` over baseline; methodologically separate from the T=0 main framework) |
| Risk-averse / clean fix/hurt (separate T>0 probe) | "2-of-K else base" on K=3 retries (`9 fixes / 2 hurts`, smaller net) |
| Compute-sensitive | Same threshold rule but tighter (e.g. `k=25`, `+1.14pt` for ~10% of compute spend) |
| Avoid | Random retry on unflagged samples (`−1.67pt ± 2.72pt`) |

On SVAMP, the analogous rule needs a SVAMP-tuned score and the **tightest budget**: wider tau adds hurts at the same rate as fixes.

These are all artifact-level numbers on a single outer split. No cross-outer-split confirmation.

---

## 7. Honest caveats

The project's claims, ranked by how confident we should be:

| Claim | Strength |
|---|---|
| The reliability score has positive AURC against random across val/test/SVAMP transfer | **Strong** |
| Selective retry on score-flagged samples beats `exp_only` on the current GSM8K outer split | **Moderate-Strong** (single retry seed, but `K=4` random controls all positive) |
| The retry mechanism rescues `~14–19%` of base-wrong flagged samples across GSM8K and SVAMP | **Moderate** (small counts, single retry per task) |
| The budget-scaling shape differs systematically between GSM8K and SVAMP | **Moderate** (clean direction, single retry artifact per task) |
| Gap signal is not vote-aggregable | **Strong** (4 independent activation paths all failed) |
| Self-consistency at K=3 majority beats K=1 by `+0.37pt` | **Weak** (single retry artifact per K seed, K-pool methodology separate) |
| Within-retry confidence predicts which K-vote is correct | **Strong negative** — it does not |
| The targeting marginal `+3.03pt ± 0.62pt` survives changing the outer val/test split | **Unknown** — not tested; deferred to robustness phase |
| Cross-task generality beyond SVAMP (e.g., Math500) | **Unknown** — not tested |
| Per-base-wrong retry rescue rate is fundamentally task-portable | **Suggestive** — based on 2 tasks |

We have been careful in the per-pass docs to keep magnitudes at the artifact level. The summary above preserves that discipline.

---

## 8. Deferred items / suggested next moves (low-priority)

Not load-bearing for the current narrative; would strengthen specific claims:

- **Phase E-Retry-OuterSeed**: re-fit score + selective retry on `2–3` different val/test splits on GSM8K. Directly tests robustness of the targeting marginal under new sample partitions. Most useful as a robustness appendix in a writeup.
- **Math500 cross-task** (base acc `~24%`): another regime to test budget-scaling and rescue-rate portability. Likely surfaces a third pattern given Math500's parser-side AWNF issues.
- **`k > 60` GSM8K budget probe**: new GPU on score-borderline samples just outside the current flagged 60 to see if the monotone growth saturates.
- **Phase E-Retry-Pool follow-ups**: explicitly out-of-scope for the T=0 narrative; revisit only if self-consistency becomes a separate goal.
- **Phase E-Diff3**: re-fit a classifier directly on `delta = y_prob − y_exp` target. The current score predicts difficulty, not method-switch utility. A delta-trained score is a stretch improvement not load-bearing — multi-seed already shows gating works via budget bounding without it.

---

## 9. What we now believe about the cgap signal (one paragraph)

The confidence-gap signal in `dLLM-MidTruth` is a **per-sample difficulty estimator**, not a per-event vote weight. As a vote weight it adds nothing on top of the temporal exp baseline (four independent attempts confirm this). As a per-sample reliability score, it isolates a slice of the test set where retrying the model is worthwhile: on the current GSM8K and SVAMP variants, roughly `~14–19%` of base-wrong flagged samples get rescued by a fresh retry, and the baseline test accuracy moves by `+2.65pt` on the current GSM8K artifact under a budget-matched random control margin of `+3.03pt ± 0.62pt`. The same mechanism works weakly on SVAMP (`+1.67pt`) when the score is task-tuned; the budget-scaling shape changes across tasks because the underlying base-accuracy distribution does. Self-consistency at `K > 1` adds only a small additional `+0.37pt` over selective retry and should be reported as a separate technique. The right operating direction going forward is sample-level **selective compute allocation** — abstention, retry, ensembling — not in-vote reweighting.

---

## 10. File index

| Topic | Primary doc |
|---|---|
| Detailed phase records | `eval/CONFIDENCE_GAP_EXPERIMENTS.md` |
| Implementation plan + checklist | `eval/CGAP_IMPROVEMENT_PLAN.md` |
| This summary | `eval/CGAP_PROJECT_SUMMARY.md` |
| Per-pass interpretations | `eval/analysis/*_interpretation.md` |
| Selective retry artifact (GSM8K T=0, k=60) | `outputs/.../20260514_gsm8k_retry_exp_testsubset_tau05845_v6/` |
| Random control artifacts (GSM8K T=0, 4 complement seeds) | `outputs/.../20260514_gsm8k_retry_exp_complement_seed{124..127}/` |
| SVAMP-tuned retry artifacts | `outputs/.../20260515_svamp_retry_exp_svamptuned_*/` |
| K-pool retry artifacts (T=0.2, seeds 42/43/44) | `outputs/.../20260515_gsm8k_retry_exp_pool_t02_seed{42,43,44}/` |
| Reliability score scripts | `eval/scripts/analyze_reliability*.py` |
| Retry pipeline + evaluators | `eval/scripts/{select_*,evaluate_*}.py` |
