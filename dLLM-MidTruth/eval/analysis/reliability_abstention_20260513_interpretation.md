# Phase E — Reliability-aware Abstention: Interpretation

**Date**: 2026-05-13
**Companion artifacts**: `reliability_abstention_20260513.{md,json}`
**Script**: `eval/scripts/analyze_reliability.py`
**Status**: **POSITIVE** — first non-trivial gain from the gap signal across the cgap project, but in a different application axis from voting.

## What this pass actually tested

Phase 8 offline proxy left one positive finding: `max_gap` correlated with sample-level baseline correctness at Pearson +0.332. That signal could not be turned into a within-sample vote weight (Phase 7/7B/7-filter/Phase 8 all null), so this pass asks the next obvious question: **can the same per-sample reliability features support a selective abstention policy?** Concretely, sort samples by a reliability score, abstain on the lowest-scoring ones, and check whether the kept samples are reliably more correct than baseline.

Score variants:
- **Single features**: each of {`max_gap`, `mean_gap`, `last_change_gap`, `first_appear_gap`, `std_gap`, `n_valid`, `n_unique_answers`, `last_change_step_frac`, `max_gap_step_frac`, `first_appear_step_frac`}, auto-oriented by val Pearson sign.
- **Combined**: val-fitted signed z-score sum across all features with `|corr| ≥ 0.05`; applied unchanged to test and SVAMP.
- **Oracle / Random**: empirical ceiling (sort by `is_correct`) and floor (random shuffle).

`is_correct` target is `exp_only` baseline correctness (the same baseline Phase 7-family was trying to beat).

## Headline numbers

### GSM8K — combined score is the consistent winner

| Coverage | Val (n=1055) | Test (n=264) | Random (val) | Oracle (val) |
|---:|---:|---:|---:|---:|
| 100% (base) | 70.62% | 67.42% | 69.55% | 70.62% |
| 80% | **78.08%** | **74.88%** | 69.55% | 88.27% |
| 50% | **86.74%** | **81.06%** | 70.08% | 100% |

- Val AURC: **0.2187** (random 0.2985, oracle 0.1234). The combined score sits about **53% of the way from random to oracle** in AURC terms.
- Test AURC: **0.2515** (random 0.3034, oracle 0.1521) — same ranking pattern survives one-shot test eval. No collapse, modest val→test gap.
- Test selective lift: SelAcc@80% is **+7.5pt** over base, SelAcc@50% is **+13.6pt** over base.

### SVAMP transfer — same val-fitted score, no refit

| Coverage | SVAMP (n=300) | Random | Oracle |
|---:|---:|---:|---:|
| 100% (base) | 86.33% | 86.25% | 86.33% |
| 80% | **91.67%** | 86.25% | 100% |
| 50% | 91.33% | 86.67% | 100% |
| Cov@90%acc | **90%** | n/a | — |

- SVAMP AURC: **0.0954** (random 0.1366, oracle 0.0298) — about 38% from random to oracle.
- We can keep **90% of SVAMP samples while maintaining ≥90% accuracy** using the val-fitted combined score. That is a non-trivial transfer result.

### Single-feature ranking

GSM8K val (AURC, low = better): `max_gap 0.231` < `last_change_gap 0.244` < `n_unique_answers 0.245` < `last_change_step_frac 0.247` < `std_gap 0.252` < others. The combined score (0.219) sits below every single feature, confirming the combination is doing real work.

SVAMP single-feature ranking partially reorders: `n_unique_answers 0.082` < `last_change_gap 0.086` < `max_gap 0.090`. Plausible reason: SVAMP base accuracy 86% leaves fewer "answer-flipping" samples, so `n_unique_answers` becomes a cleaner outlier indicator than `max_gap` magnitude. Combined still works well (0.095) but a single feature can edge it on this task.

## Why this works when Phase 7-family didn't

The same gap data feeds both, but the signal is being read on different axes:

| Axis | What is being asked | What the signal supports | Vote pipeline result |
|---|---|---|---|
| Within-sample event ranking (Phase 3) | Which event in a sample has the correct answer? | win-rate 0.86 | Not vote-aggregable; Phase 7-family null |
| Sample-level reliability (Phase 8 proxy + Phase E) | Which samples is the model likely to get right? | max_gap +0.33, combined AURC bin between random/oracle | **Drives selective abstention; this pass** |

Voting consumes per-event sums and the temporal mass saturates at one settlement step, so within-sample ranking has no room to lift the vote. Abstention consumes per-sample scores and asks a different question (keep or drop the sample), where the same features are directly actionable.

## Caveats and limits

- **Combined score is correlation-sign z-score sum, not a fitted classifier.** A logistic regression / gradient-boosted classifier could likely do better; this is a deliberately simple baseline to avoid overfitting on n=1055 val.
- **Math500 not tested here.** Its valid-event population differs sharply (33.6% AWNF, many "no boxed" generations), and several features (`n_unique_answers`, `last_change_step_frac`) would have a degenerate distribution on Bucket-A-heavy samples. Worth a separate pass that treats Math500 explicitly rather than transferring the GSM8K score.
- **Target is `exp_only` correctness.** Using a different voting method as the baseline (e.g. `cgap_rawsum`, which beat `exp_only` on test 68.94% vs 67.42%) would shift the reliability label slightly. Effect likely small but unverified.
- **Cov@90%acc = n/a for GSM8K.** Even oracle reaches 88.27% at 80% coverage on val and ~84% on test — base accuracy 70% is too low for any score to cross 90% selective accuracy without abstaining extremely aggressively. SVAMP's higher base lets it cross.
- **No GPU rerun was needed.** This entire pass is offline post-processing on existing debug data.

## What this opens up

This is the first pass in the cgap project where the gap signal turns into a measurable downstream win. Concrete follow-up directions, in rough order of cheapest-to-most-ambitious:

- **Phase E2 — Math500 reliability pass.** Same script with task-specific feature handling for Bucket-A-heavy samples. Likely needs a `parse_failed`-aware feature (e.g. fraction of steps that produced any valid parse).
- **Phase E3 — Better combined score.** Logistic regression on val features; honest comparison vs the simple z-score sum.
- **Phase E4 — Abstention-as-retry.** When the score is low, regenerate with a different seed / different decoding params and re-vote across both runs. This converts abstention into an action.
- **Phase E5 — Calibration plots.** Bin combined score, plot empirical accuracy per bin. Useful for productizing a confidence number rather than just a ranking.

For the plan/tracker, this changes the picture meaningfully: the gap signal is **no longer dead** in this project, it just lives outside the voting pipeline.
