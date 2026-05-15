# cgap Project Summary

**Date**: 2026-05-15  
**Scope**: confidence-gap voting, reliability scoring, and archived rerun probes for `dLLM-MidTruth`  
**Companion docs**: detailed phase records in `CONFIDENCE_GAP_EXPERIMENTS.md`, implementation plan in `CGAP_IMPROVEMENT_PLAN.md`, per-pass interpretations in `eval/analysis/*_interpretation.md`

This document is a consolidated summary of what the cgap project **actually established**, what it **ruled out**, and which later rerun probes should now be treated as **archived exploratory side paths rather than evidence**.

---

## 1. Project arc — original goal and what shifted

**Original goal**: test whether confidence-gap signals can replace or complement the time-based `exp` voting weights from dLLM-MidTruth's TSCV.

**What survived**:

1. **Gap as vote weight** did not work.
2. **Gap-derived reliability / difficulty scoring** did work.
3. **T=0 rerun-based retry claims do not hold up as method evidence** and are now archived.
4. **T>0 K-pool** is a separate self-consistency probe, not a continuation of the main cgap line.

The project therefore converged on a narrower but cleaner statement:

> The confidence-gap signal is useful as a **sample-level reliability / difficulty score**, not as an in-vote weight.

---

## 2. Layer 1 — gap as vote weight: ruled out

The original hypothesis was that a gap-aware per-step vote should beat `exp_only`.

| Pass | Mechanism | Result |
|---|---|---|
| Phase 7 | additive / multiplicative hybrid with `exp_weight + λ·quality(gap)` | no GSM8K test lift |
| Phase 7B | exp normalization + wider `λ` | val moved, test still no lift |
| Phase 7-filter | prune low-gap events before voting | best result was effectively “no filtering” |
| Phase 8 proxy | commit-equivalent single-event proxy vote | recovers `~99%` of `exp_only`; voting headroom already tiny |

**Conclusion**: on the current artifacts, the confidence-gap signal is **not vote-aggregable** in a way that improves over `exp_only`.

This is the cleanest negative in the project: several distinct activation paths all failed in the same direction.

---

## 3. Layer 2 — reliability scoring: the load-bearing positive

The useful pivot was:

> even if gap does not help *inside* a vote, it may still say which **samples** are hard or unreliable.

### 3.1 Sample-level abstention works

Per-sample features such as `max_gap`, `n_unique_answers`, and `last_change_step_frac` produce a real reliability ranking.

| Strategy | GSM8K test AURC | random | oracle |
|---|---:|---:|---:|
| zsum_combined | 0.2515 | 0.3034 | 0.1521 |
| logistic_broad | **0.2354** | 0.3034 | 0.1521 |

This is a strong positive. The score is clearly better than random at ranking which samples are likely to be correct.

### 3.2 Offline fallback gating is small but real

The cleanest operational result that stays inside the canonical `T=0` framework is the **offline fallback gate**:

- rule: `if logistic_broad_score < tau: prob_vote else exp_only`
- after the multi-seed pass, this should be read as a **small mean gain**, not a big single-seed one
- GSM8K 10-seed summary:
  - `gated − exp_only = +0.61pt ± 0.78pt`
  - `gated − prob_vote = +0.72pt ± 1.11pt`

The key interpretation is:

- the score is useful enough to support a **budget-bounded fallback swap**
- but it is **not** a strong predictor of which alternate answer source will beat `exp_only`

### 3.3 What the score predicts — and what it does not

The differential follow-up made this especially clear:

- `Pearson(score, y_exp)` is positive and substantial
- `Pearson(score, delta = y_prob - y_exp)` is near zero

So the score is best understood as a **difficulty / reliability score**, not a **method-switch utility score**.

### 3.4 Cross-task read

What carries across tasks most cleanly is the **reliability ranking** itself:

- GSM8K: strong abstention/ranking signal
- SVAMP: weaker but still useful reliability signal, including transfer on abstention-style metrics
- Math500: weaker signal again, with parser-side issues dominating much of the task behavior

This supports a modest but defensible cross-task claim:

> confidence-gap-derived features are useful for **sample-level reliability assessment**, but their strength is task-dependent.

---

## 4. Math500 read — parser-side bottleneck, not a cgap voting win

Math500 was useful mainly as a diagnosis task.

What we learned:

- large AWNF was **not** mainly a char-offset localization issue
- parser-side Bucket A dominated
- conservative formatting rescue recovered many parseable cases but barely moved accuracy

So Math500 did **not** turn into a confidence-gap success story. It instead clarified that parser/format robustness is a separate bottleneck.

---

## 5. Archived rerun probes — keep the files, drop the claims

Several later analyses used `T=0` rerun artifacts and treated them as if “retrying” a low-score sample produced a meaningful fresh second attempt.

We no longer treat those results as load-bearing.

### Why

Under a truly identical `T=0` setup:

- no temperature noise is added
- decoding is `argmax`
- same seed / same model / same inputs / same runtime path should produce the same answer

So a `T=0` rerun should be interpreted as deterministic unless the setup changed.

In our rerun artifacts, the setup was **not fully identical** to the original base artifact:

- stored `batch_size` differs
- stored `model_path` differs
- stored `vote_method` / run metadata differ
- not all old artifacts persisted enough metadata (notably seed) to certify identicality end-to-end

Because of that, the rerun comparisons are best read as:

> artifact-level rerun probes under non-identical conditions

not as evidence for a clean retry mechanism.

### What this means for the project

The following lines are now **archived exploratory only**:

- `T=0` “true selective retry”
- random-budget rerun controls built on that same rerun setup
- rerun-based cross-task comparisons
- rerun-based budget sweeps

They may still be interesting as artifact probes, but they are **not** part of the main claim set anymore.

---

## 6. K-pool / self-consistency probe — separate and weak

The `T>0` K-pool branch is conceptually different.

It asks:

> if we deliberately inject sampling diversity, does self-consistency help?

That is a reasonable side question, but it is **not the same question** as the main cgap project.

What remains from that branch:

- diversity exists
- naive confidence-based selectors do **not** identify the correct retry well
- `K=3 majority` gives only a small gain over `K=1`

So K-pool should be reported, if at all, as a **separate self-consistency probe with a mostly negative mechanism finding**.

---

## 7. What we can honestly claim now

### Strong

- Gap-derived weights do **not** improve over `exp_only` voting on the current artifacts.
- Confidence-gap-derived features provide a real **sample-level reliability ranking**.
- The strongest positive evidence is in abstention / calibration / ranking, not in vote replacement.

### Moderate

- A simple offline fallback gate (`score < tau -> prob_vote`) gives a **small mean gain** on GSM8K under multi-seed evaluation.
- The score behaves like a **difficulty signal**, not a method-switch utility signal.

### Weak / archived

- Any apparent `T=0 rerun` lift
- Any rerun-based targeting marginal
- Any rerun-based cross-task scaling story

Those are no longer part of the main evidence set.

---

## 8. Recommended interpretation going forward

The right way to describe the cgap signal now is:

> The confidence-gap signal in `dLLM-MidTruth` is best understood as a **per-sample reliability / difficulty estimator**. It does not improve the internal vote, but it does help rank which samples are trustworthy and which are not.

That means the natural application directions are:

- abstention
- calibration
- confidence-aware fallback gating
- selective compute allocation in settings where the additional computation is genuinely stochastic or otherwise meaningfully different

What it does **not** currently support is:

- stronger vote weights
- deterministic `T=0` rerun claims

---

## 9. File index

| Topic | Primary doc |
|---|---|
| Detailed phase records | `eval/CONFIDENCE_GAP_EXPERIMENTS.md` |
| Implementation plan + checklist | `eval/CGAP_IMPROVEMENT_PLAN.md` |
| This summary | `eval/CGAP_PROJECT_SUMMARY.md` |
| Per-pass interpretations | `eval/analysis/*_interpretation.md` |
| Reliability score scripts | `eval/scripts/analyze_reliability*.py` |
| Offline gating / fallback evaluators | `eval/scripts/{evaluate_true_retry.py,evaluate_retry_budget_sweep.py,select_*,evaluate_*}.py` |

> Historical note: rerun / retry / K-pool artifacts are kept on disk for traceability, but they should now be read as archived exploratory probes rather than as load-bearing evidence.
