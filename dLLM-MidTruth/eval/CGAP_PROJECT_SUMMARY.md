# cgap Project Summary

**Date**: 2026-05-19  
**Scope**: confidence-gap voting, reliability scoring, archived rerun probes, and token-ordering ablations for `dLLM-MidTruth`  
**Companion docs**: detailed phase records in `CONFIDENCE_GAP_EXPERIMENTS.md`, implementation plan in `CGAP_IMPROVEMENT_PLAN.md`, token-ordering line in `TOKEN_ORDERING_EXPERIMENT.md`, transfer-score results in `eval/analysis/transfer_score_ablation_full_20260516.md`, per-pass interpretations in `eval/analysis/*_interpretation.md`

This document is a consolidated summary of what the cgap project **actually established**, what it **ruled out**, and which later rerun probes should now be treated as **archived exploratory side paths rather than evidence**.

---

## 1. Project arc — original goal and what shifted

**Original goal**: test whether confidence-gap signals can replace or complement the time-based `exp` voting weights from dLLM-MidTruth's TSCV.

**What survived**:

1. **Gap as vote weight** did not work.
2. **Gap-derived reliability / difficulty scoring** did work.
3. **T=0 rerun-based retry claims do not hold up as method evidence** and are now archived.
4. **T>0 K-pool** is a separate self-consistency probe, not a continuation of the main cgap line.
5. A new **token-ordering line** does produce a clean raw-accuracy lift.

The project therefore converged on a narrower but cleaner statement:

> The confidence-gap signal is useful as a **sample-level reliability / difficulty score**, not as an in-vote weight.
>
> Separately, a **decoder-level token-ordering change** does create real headroom even after answer-level voting ideas saturated.

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

### 3.5 Within-artifact routing / coalition probes also saturate

We also checked the last remaining methodology-clean raw-accuracy angle:

> inside a **single T=0 base artifact**, can we derive multiple deterministic answer candidates and route among them better than `exp_only`?

This line stayed fully inside the strict setup:

- same `T=0` artifact
- same parser / evaluator
- no rerun
- no extra inference

The candidate pool included:

- `exp_only`
- native stored `vote_answer`
- `final_answer`
- temporal readouts such as `last_valid_answer`, `late_window_majority_q25/q50`, `longest_run_answer`, and `most_persistent_answer`

What we found:

- **Val-selected single-readout swap = `+0.00pt`**
  - best val readout was still `exp_only` (`70.62%`)
  - the test deployment therefore stayed at baseline (`67.42%`)
- **Val-selected conservative coalition override = `+0.00pt`**
  - the best val rule was effectively “do nothing”
  - test produced `0` overrides, `0` fixes, `0` hurts
- **Oracle ceiling exists but is not capturable by simple routing**
  - full oracle across within-artifact candidates: `70.83%` (`+3.41pt`)
  - compact oracle (`exp_only` / native vote / final answer): `68.94%`
  - temporal candidates therefore add only `+1.89pt` of extra theoretical headroom
  - all of this headroom sits inside a small disagreement slice (`33/264 = 12.5%`)

The important interpretation is:

> even when we stay entirely inside one clean deterministic artifact, there is no val-stable signal for selecting a better answer than `exp_only`.

So the raw-accuracy line is not only saturated at the vote-weight level; it is also saturated for:

- single-readout replacement
- conservative within-artifact routing
- simple temporal coalition rules

This is consistent with the earlier differential result: the score tracks **difficulty**, not **method-switch / routing utility**.

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

## 7. Layer 4 — token ordering: first clean raw-accuracy lift

The strongest new positive result in the project comes from a different place in the pipeline:

> not answer-level vote weighting, but **which masked token positions get opened first** during deterministic decoding.

This line stays methodology-clean:

- same `T=0` deterministic setup
- same parser
- same prompt
- same model checkpoint
- same generation length / diffusion steps / block length
- same final `exp`-weighted TSCV vote
- same seed
- no rerun / no setup mismatch

The only change is the transfer ranking score used inside the active block:

- baseline: `score_i = p_top1(i)`
- ablation: `score_i = p_top1(i) - p_top2(i)`

This is a decoder-policy change, not a voting-method change.

### Full-run result

| Task | Baseline vote | `prob_margin` vote | delta | Baseline final | `prob_margin` final | delta |
|---|---:|---:|---:|---:|---:|---:|
| GSM8K | 69.67% | 70.81% | `+1.14pt` | 68.39% | 69.37% | `+0.98pt` |
| SVAMP | 86.00% | 87.33% | `+1.33pt` | 84.33% | 86.67% | `+2.34pt` |
| MATH500 | 27.60% | 27.60% | `+0.00pt` | 27.00% | 27.20% | `+0.20pt` |
| Countdown | 23.05% | 23.05% | `+0.00pt` | 19.53% | 18.36% | `-1.17pt` |

### Read

- **GSM8K** and **SVAMP** show a real positive lift on the primary metric (`vote_answer` under `exp` TSCV).
- **MATH500** and **Countdown** do not.
- The pattern is consistent with the earlier project diagnosis:
  - parser-clean tasks benefit
  - parser-bottlenecked tasks do not convert the changed trajectory into better measured accuracy

This is the first result in the project that is:

- raw-accuracy positive
- deterministic
- parser-fixed
- vote-fixed
- setup-clean

So the project narrative now changes from:

> answer-level voting ideas are saturated

to:

> answer-level voting ideas are saturated, but **token-level decoding order still has real headroom**.

### First temporal extension status

The first temporal follow-up on top of `prob_margin` was:

- `temporal_margin = prob_margin + λ · run_length_stability`

On a `GSM8K` smoke subset (`n=64`), this first `C1` extension did **not** beat plain `prob_margin`:

- `A = top1_prob`: vote `76.56%`
- `B = prob_margin`: vote **`79.69%`**
- `C1a (λ=0.05)`: vote `75.00%`
- `C1b (λ=0.10)`: vote `76.56%`
- `C1c (λ=0.20)`: vote `75.00%`

So the current honest state of Layer 4 is:

- `prob_margin` is the best current rule
- the first naive temporal extension is negative
- any further temporal claim now depends on redesign, not on simply scaling up `C1`

A short mechanism read from the overlap pass is:

- `prob_margin` gains come from broad **early** transfer-order reordering relative to `top1_prob`
- `C1` also changes the order in a real way, but mostly in a direction that erodes `prob_margin`'s gain rather than extending it

### Gated temporal follow-up

We also ran a first redesigned temporal follow-up at full scale:

- baseline: `prob_margin`
- candidate: `gated_temporal_margin = margin + λ · stability · 1[margin < τ]`
- settings: `λ = 0.10`, `τ = 0.15`

Result vs `prob_margin`:

| Task | `prob_margin` vote | `gated_temporal_margin` vote | delta |
|---|---:|---:|---:|
| GSM8K | 70.81% | 70.58% | `-0.23pt` |
| SVAMP | 87.33% | 88.67% | `+1.34pt` |
| MATH500 | 27.60% | 28.40% | `+0.80pt` |
| Countdown | 23.05% | 23.44% | `+0.39pt` |

This is a useful refinement of the Layer-4 story:

- the temporal signal is **not** uniformly harmful
- but the first gated redesign is still **not** a clean replacement for `prob_margin`, because it gives up GSM8K to gain on the other three tasks

So the current honest ranking is:

1. `prob_margin` remains the best current default for the original Layer-4 claim
2. `gated_temporal_margin` is a promising task-dependent variant
3. naive `C1` remains a negative first temporal extension

### What is novel here

This is **not** a claim that probability-margin token ordering itself is a new idea; that baseline comes from the Kim et al. token-ordering line.

The novelty of the current result is narrower and more honest:

- we applied a Kim-style token-ordering intervention inside the current `LLaDA + exp-TSCV` math-style decoding setup
- we showed that it produces a clean gain exactly where the earlier answer-level cgap ideas had saturated
- we now have a principled next extension: a **temporal-margin** score that combines Kim-style local margin with the temporal-stability viewpoint from `Time Is a Feature` / `Prophet`

So the current Layer 4 result should be read as:

> a strong new baseline for this project's next phase, and the first clean accuracy lift discovered here.

---

## 8. What we can honestly claim now

### Strong

- Gap-derived weights do **not** improve over `exp_only` voting on the current artifacts.
- Confidence-gap-derived features provide a real **sample-level reliability ranking**.
- The strongest positive evidence is in abstention / calibration / ranking, not in vote replacement.
- A decoder-level token-ordering change (`top1_prob -> prob_margin`) gives a **clean raw-accuracy lift** on GSM8K and SVAMP with all other controls fixed.

### Moderate

- A simple offline fallback gate (`score < tau -> prob_vote`) gives a **small mean gain** on GSM8K under multi-seed evaluation.
- The score behaves like a **difficulty signal**, not a method-switch utility signal.

### Weak / archived

- Any apparent `T=0 rerun` lift
- Any rerun-based targeting marginal
- Any rerun-based cross-task scaling story

Those are no longer part of the main evidence set.

### Null but important

- Even with a methodology-clean within-artifact candidate router, **val-selected raw-accuracy improvement stays at `+0.00pt`**
- Oracle headroom exists, but it lives in a small disagreement slice and is not captured by simple deterministic routing rules
- Answer-level routing / coalition ideas remain saturated even after a clean within-artifact pass, which makes the Layer 4 token-ordering gain more informative

---

## 9. Recommended interpretation going forward

The right way to describe the cgap signal now is:

> The confidence-gap signal in `dLLM-MidTruth` is best understood as a **per-sample reliability / difficulty estimator**. It does not improve the internal vote, but it does help rank which samples are trustworthy and which are not.

That means the natural application directions are:

- abstention
- calibration
- confidence-aware fallback gating
- selective compute allocation in settings where the additional computation is genuinely stochastic or otherwise meaningfully different

The right way to describe the new accuracy result is:

> the project did **not** find a better answer-level vote, but it **did** find a better token-ordering policy inside deterministic masked-diffusion decoding.

What it does **not** currently support is:

- stronger vote weights
- deterministic `T=0` rerun claims
- a robust raw-accuracy lift through within-artifact answer routing

The most natural next step is therefore not another answer router, but a **temporal-margin token-ordering extension**.

---

## 10. File index

| Topic | Primary doc |
|---|---|
| Detailed phase records | `eval/CONFIDENCE_GAP_EXPERIMENTS.md` |
| Implementation plan + checklist | `eval/CGAP_IMPROVEMENT_PLAN.md` |
| Token-ordering experiment line | `eval/TOKEN_ORDERING_EXPERIMENT.md` |
| Transfer-score A/B results | `eval/analysis/transfer_score_ablation_full_20260516.md` |
| This summary | `eval/CGAP_PROJECT_SUMMARY.md` |
| Per-pass interpretations | `eval/analysis/*_interpretation.md` |
| Reliability score scripts | `eval/scripts/analyze_reliability*.py` |
| Offline gating / fallback evaluators | `eval/scripts/{evaluate_true_retry.py,evaluate_retry_budget_sweep.py,select_*,evaluate_*}.py` |

> Historical note: rerun / retry / K-pool artifacts are kept on disk for traceability, but they should now be read as archived exploratory probes rather than as load-bearing evidence.
