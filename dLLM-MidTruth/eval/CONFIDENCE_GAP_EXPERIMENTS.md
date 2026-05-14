# Confidence Gap Experiment Tracker

This file tracks the confidence-gap voting experiments for `dLLM-MidTruth`.

## Goal

- Reproduce the original TSCV (`exp`) results.
- Evaluate whether confidence-gap signals can replace or complement time-based voting weights in dLLM voting.
- Keep baselines, variants, and future runs in one place.

## Paper References

### Paper Base / Exp

| Task | Paper Base | Paper Exp |
|---|---:|---:|
| Countdown | 20.3 | 25.0 |
| GSM8K | 68.5 | 70.1 |
| MATH500 | 27.4 | 28.4 |
| SVAMP | 84.0 | 86.0 |

### Paper Ever-Pass

| Task | Measured Ever-Pass | Paper Ever-Pass |
|---|---:|---:|
| Countdown | 29.30 | 28.1 |
| GSM8K | 80.82 | 80.5 |
| MATH500 | 41.20 | 40.2 |
| SVAMP | 91.00 | 91.3 |

## Methods

- `exp`
  - Original dLLM-MidTruth temporal voting baseline.
- `confidence_gap_answer_window5_mean_rawsum` (logit)
  - Parsed-answer locate variant using the entire `window5`. Raw logit gap.
- `confidence_gap_answer_window5_prob_mean_rawsum` (prob) ← **current mainline**
  - Same locate strategy, probability gap instead of raw logit gap.
  - Compresses per-step weights to [0, 1]; reduces extreme outlier dominance.
  - Slight gain over logit on Math500 (+1.0p vote acc) and SVAMP (+0.34p vote acc).
- `confidence_gap_answer_window5_mean_rawsum_skip33pct`
  - Same as logit variant, but skips the first 33% of steps. No material improvement.
- `confidence_gap_answer_window5_blockactive_prob_mean_rawsum`
  - Blockactive/prob-gap variant. Discards too many valid answer events in practice.
- `confidence_gap_anchor_window5_logit_mean_rawsum` ← **archived exploratory**
  - Strict Prophet-compatible anchor. Uses `constraints_text="96:The answer is"`.
  - Modifies the generation trajectory itself, not just vote weights.
  - Hurts Countdown severely; does not beat `exp` on any task. See Strict Anchor Debug Findings.

## Current Main Results (bs4, gen128, steps64)

| Task | Final | Exp Vote | CGap Logit Vote | CGap Skip33 Vote | CGap Prob Vote | CGap Blockactive Prob Vote |
|---|---:|---:|---:|---:|---:|---:|
| Countdown | 21.48 | 25.39 | 24.22 | 24.22 | 23.05 | 16.80 |
| GSM8K | 68.69 | 69.98 | 69.83 | 69.83 | 69.67 | 69.07 |
| MATH500 | 27.00 | 27.20 | 24.60 | 24.60 | **25.60** | 23.80 |
| SVAMP | 84.67 | 86.33 | 86.33 | 85.67 | **86.67** | 84.67 |

## Current Read on the Results

- In the current runs, `exp` remains the strongest overall reference point across the four tasks.
- `confidence_gap_answer_window5_prob_mean_rawsum` (prob) slightly outperforms the logit variant on Math500 (+1.0p vote acc) and SVAMP (+0.34p vote acc), which suggests probability normalization may help where raw logit outliers dominate.
- Neither locate variant beats `exp` overall so far; on Math500, high AWNF rate (33.6%) remains a plausible bottleneck.
- `skip33` does not appear to help materially in the current setting.
- `blockactive_prob` is the most principled variant so far, but in practice it discards many valid answer events:
  - many correct events are skipped as `answer_window_outside_active_block`
  - this removes the repeated-consistency signal that `exp` benefits from
- The current mainline direction is therefore less about a pure `exp` replacement and more about:
  - repairing localization failures
  - testing whether confidence can complement temporal voting more effectively
- A follow-up Math500 Bucket A diagnosis suggests the current parser-side bottleneck is not a single narrow bug:
  - many events never reach any recoverable boxed span
  - another large slice contains malformed or repeated `boxed...` formatting
  - this makes a pure char-offset repair look less central than it first appeared
- An offline Bucket A rescue upper bound points in the same direction:
  - conservative formatting-only heuristics can make many Bucket A events parseable again
  - but only a small fraction become GT-equivalent, and the sample-level `exp_only` upper bound moves only slightly
  - so parser-side hardening still looks more like a secondary robustness path than a primary explanation for the current Math500 ceiling
- The clearest current operational rule is now in the reliability track:
  - `if logistic_broad_score < 0.5845 -> prob_vote else keep exp_only`
  - on the current artifact this gives GSM8K `67.42% → 68.94%` (`+1.52pt`) and SVAMP `86.33% → 86.67%` (`+0.33pt`)

## Phase 3 Status

- `2026-05-12`: initial GSM8K gap-correctness analysis completed from
  [gsm8k_gap_corr_2026-05-12.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/gsm8k_gap_corr_2026-05-12.md)
  and
  [gsm8k_gap_corr_2026-05-12.json](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/gsm8k_gap_corr_2026-05-12.json).
- Despite the run directory name `20260429_everpass_debug_bs4`, the stored `vote_method` inside the JSON is `confidence_gap_answer_window5_mean_rawsum`, and all valid GSM8K events in that artifact include `gap_values`.
- Preliminary read:
  - unconditional correlation is positive (`Pearson 0.2201`, `Spearman 0.2338`)
  - within-sample pairwise win-rate is well above random
    - raw: `0.7625`
    - contiguous dedup: `0.7034`
    - answer-level unique: `0.8632`
- This is encouraging enough to keep the hybrid line alive, but it is still a pre-patch signal check. Final λ/quality selection should still wait for the patched GSM8K debug data from Phase 5.5.
- **Update (2026-05-13)**: subsequent Phase 7/7B/7-filter and Phase 8 offline proxy passes turned this positive within-sample ranking signal into no observable vote gain across four activation paths. See "Hybrid line closure" below.

## Phase 7 Status

- `2026-05-12`: initial GSM8K hybrid sweep completed from
  [gsm8k_hybrid_sweep_20260512.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/gsm8k_hybrid_sweep_20260512.md)
  and
  [gsm8k_hybrid_sweep_20260512.json](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/gsm8k_hybrid_sweep_20260512.json).
- This was an **exploratory pre-patch sweep** on the existing GSM8K artifact, not the planned patched-GSM8K main pass.
- Best validation config in that sweep:
  - quality shape: `clipped`
  - `λ = 0.25`
  - formula: `multiplicative`
- Readout:
  - validation: very small edge over `exp_only`
  - one-shot GSM8K test: `hybrid(best)` matched `exp_only` and stayed below `cgap_rawsum`
  - SVAMP transfer: no observable gain over either baseline
- Current read:
  - the sweep does **not** provide strong evidence for a useful hybrid yet
  - but it also does not fully close the line, because it was run on pre-patch data and with a lightweight exploratory protocol
  - one plausible failure mode is scale mismatch: late-step `exp` weights can dominate the current `quality × λ` range, so the hybrid may collapse back toward `exp`
  - the next main decision point should come either after patched GSM8K debug data from Phase 5.5, or after explicitly deciding that current GSM8K data is a good enough patched-equivalent reference
  - **Update (2026-05-13)**: superseded by the closure below — Phase 7-filter and Phase 8 offline proxy probed the gap signal directly without relying on patch quality, so the Phase 5.5 gate is no longer a meaningful condition for this line

- `2026-05-12`: a follow-up `Phase 7B` exploratory pass with exp scaling + wider `λ` also completed from
  [gsm8k_hybrid_sweep_v2_20260512.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/gsm8k_hybrid_sweep_v2_20260512.md)
  and
  [gsm8k_hybrid_sweep_v2_20260512.json](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/gsm8k_hybrid_sweep_v2_20260512.json).
- Best validation config in that v2 sweep:
  - exp mode: `max_norm`
  - quality shape: `clipped`
  - `λ = 2.0`
  - formula: `multiplicative`
- Readout:
  - validation improved a bit more (`71.00%`)
  - one-shot GSM8K test still matched `exp_only` (`67.42%`) and stayed below `cgap_rawsum`
  - SVAMP transfer again showed no gain
- Current read:
  - simple exp-scale normalization seems to reduce the validation-side collapse a little
  - but it still does **not** give convincing test-time or transfer gains in the current exploratory setting
  - this makes it less likely that raw exp scale was the only bottleneck

- `2026-05-13`: a `Phase 7C`-style exploratory pass using **gap-as-filter** also completed from
  [gsm8k_gap_filter_sweep_20260513.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/gsm8k_gap_filter_sweep_20260513.md)
  and
  [gsm8k_gap_filter_sweep_20260513.json](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/gsm8k_gap_filter_sweep_20260513.json).
- Setup:
  - per-sample gap quantile filter (`0/25/50/75`)
  - surviving events are still accumulated with the original `exp` temporal weight
  - matched-count `late_control` used as a control
- Readout:
  - validation best quantile was `0`, i.e. no filtering
  - one-shot GSM8K test stayed at `67.42%`, matching `exp_only`
  - SVAMP transfer also showed no gain
  - additional structural observation: at every non-zero quantile, `late_control` matched or beat `gap_filter` on validation (`70.71%` vs `70.43–70.62%`), so step-based pruning is at least as good as gap-based pruning
- Current read:
  - in the current artifact, the Phase 3 ranking signal does **not** seem to turn into an obvious gain through simple pruning/gating
  - together with Phase 7/7B, this makes it less likely that a lightweight hybrid/filter formulation is enough by itself
  - the `late_control ≥ gap_filter` pattern is additional evidence that gap and step signals are largely overlapping for pruning purposes

## Phase 8 Status (offline proxy)

- `2026-05-13`: offline commit-gap proxy completed from
  [gsm8k_commit_gap_proxy_20260513.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/gsm8k_commit_gap_proxy_20260513.md),
  [gsm8k_commit_gap_proxy_20260513.json](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/gsm8k_commit_gap_proxy_20260513.json),
  and interpretation notes
  [gsm8k_commit_gap_proxy_20260513_interpretation.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/gsm8k_commit_gap_proxy_20260513_interpretation.md).
- This is an **offline proxy** of the planned commit-time hook. The true hook (top1-top2 gap at `select_indices` commit moment) requires a `generate.py` flag-gated change and a GPU rerun; the proxy substitutes commit-equivalent steps inferred from the existing `parsed_answer` trajectory (`first_appear`, `last_change`, `max_gap`).
- Setup:
  - per-sample single-event vote at each proxy step, compared with `exp_only` baseline
  - per-sample feature correlation with baseline (exp_only) correctness on full GSM8K (`n=1319`)
- Readout (single-event proxy vote):
  - `proxy_last_change` and `proxy_max_gap` reach `~99%` of `exp_only` accuracy using one event per sample (val `69.86–69.95%`, test `67.05–67.80%`, SVAMP `86.00%`)
  - i.e. one commit-equivalent step's answer essentially saturates the full temporal vote — this suggests the voting machinery itself has limited headroom in the current artifact, which is consistent with all earlier hybrid/filter null results
- Readout (per-sample correlation with baseline correctness):
  - `max_gap` Pearson `+0.332`, Spearman `+0.332`
  - `mean_gap` Pearson `+0.100`, Spearman `+0.115`
  - i.e. sample-level peak gap is `~3x` stronger as a between-sample reliability indicator than the diffuse temporal mean
  - structural difficulty proxies also show up: `last_change_step_frac` Pearson `−0.308`, `n_unique_answers` Pearson `−0.277`
- Current read:
  - this proxy serves as an offline upper bound on what a true commit-gap hook could buy in voting; nothing here suggests a GPU rerun would lift vote accuracy above `exp_only`
  - the new finding `max_gap` corr `+0.332` is a **between-sample** reliability signal, not a within-sample vote-ranking signal — it does not fit the voting integration shape that Phase 7/7B/7-filter were testing
  - we treat this as the **final exploratory pass for the hybrid line on the current artifact**, not as a green light to instrument the hook

## Phase E Status (sample-level reliability for abstention) — first positive result

- `2026-05-13`: reliability-aware abstention exploratory pass completed from
  [reliability_abstention_20260513.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/reliability_abstention_20260513.md),
  [reliability_abstention_20260513.json](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/reliability_abstention_20260513.json),
  and interpretation notes
  [reliability_abstention_20260513_interpretation.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/reliability_abstention_20260513_interpretation.md).
- Setup:
  - per-sample features (`max_gap`, `n_unique_answers`, `last_change_step_frac`, `std_gap`, `n_valid`, etc.)
  - target: `exp_only` baseline correctness
  - score variants: 10 single features, combined val-fitted signed z-score sum, oracle, random
  - protocol: val/test split `seed=42`, `frac=0.8` (matches Phase 7); score fit on val, applied unchanged to test and SVAMP
- Readout (combined score):
  - GSM8K val: AURC `0.2187` (random `0.2985`, oracle `0.1234`); SelAcc@80% `78.08%` (`+7.5pt` over base), SelAcc@50% `86.74%` (`+16.1pt`)
  - GSM8K test (one-shot): AURC `0.2515` (random `0.3034`, oracle `0.1521`); SelAcc@80% `74.88%` (`+7.5pt`), SelAcc@50% `81.06%` (`+13.6pt`)
  - SVAMP transfer: AURC `0.0954` (random `0.1366`, oracle `0.0298`); SelAcc@80% `91.67%` (`+5.3pt`); Cov@90%acc `90%`
- Single-feature top performers:
  - GSM8K: `max_gap` (AURC `0.231` val / `0.263` test)
  - SVAMP: `n_unique_answers` (AURC `0.082`)
  - Combined score beats every single feature on GSM8K, ties or trails the best single on SVAMP
- Current read:
  - this is the first pass in the cgap project where the gap signal produces a measurable downstream win
  - the gap signal is **not vote-aggregable** (Phase 7-family null) **but is sample-level reliability-actionable**
  - val→test transfer works without overfitting; val-fitted score also transfers to SVAMP without refitting
  - hybrid line remains closed; this is a separate Phase E track using the same data on a different application axis
- Caveats:
  - target is `exp_only` correctness; different vote method would shift labels slightly
  - Math500 not included — Bucket-A-heavy event population needs task-specific handling
  - combined score is a deliberately simple z-score sum, not a fitted classifier — a real classifier likely does better

- `2026-05-13`: an E2 follow-up with a learned combined score also completed from
  [reliability_abstention_v2_20260513.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/reliability_abstention_v2_20260513.md)
  and
  [reliability_abstention_v2_20260513.json](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/reliability_abstention_v2_20260513.json).
- Setup:
  - same GSM8K / SVAMP artifacts and outer split as Phase E
  - compare the original `zsum_combined` against two val-fitted NumPy logistic scores:
    - `logistic_top3`: `max_gap`, `n_unique_answers`, `last_change_step_frac`
    - `logistic_broad`: the broader feature subset retained by the Phase E z-score spec
  - L2 strength chosen on an inner dev split of the GSM8K validation set
- Readout:
  - GSM8K test:
    - `zsum_combined`: AURC `0.2515`, SelAcc@80% `74.88%`
    - `logistic_top3`: AURC `0.2408`, SelAcc@80% `75.36%`
    - `logistic_broad`: AURC `0.2354`, SelAcc@80% `77.73%`
  - SVAMP transfer:
    - `zsum_combined`: AURC `0.0954`, SelAcc@80% `91.67%`
    - `logistic_top3`: AURC `0.0802`, SelAcc@80% `92.92%`
    - `logistic_broad`: AURC `0.0848`, SelAcc@80% `92.50%`
- Current read:
  - a modest learned score appears to improve on the simple z-score baseline without collapsing on one-shot GSM8K or SVAMP transfer
  - this makes the reliability / abstention line look less like a one-off positive and more like a viable follow-up track
  - Math500 was still missing at this point because its Bucket-A-heavy population likely needs different feature handling

- `2026-05-14`: a task-specific Math500 reliability pass also completed from
  [math500_reliability_abstention_20260514.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/math500_reliability_abstention_20260514.md)
  and
  [math500_reliability_abstention_20260514.json](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/math500_reliability_abstention_20260514.json).
- Setup:
  - same abstention framing as Phase E, but with Math500-specific features that include AWNF / Bucket-A structure (`bucket_a_ratio`, `awnf_ratio`, `n_valid`, `last_change_step_frac`, etc.)
  - target remains offline `exp_only` correctness reconstructed from the current Math500 artifact
- Readout:
  - Math500 test base acc: `21.00%`
  - `zsum_combined`: AURC `0.7560`, SelAcc@80% `22.50%`, SelAcc@50% `32.00%`
  - best single (`n_valid`): AURC `0.7628`, SelAcc@80% `22.50%`, SelAcc@50% `28.00%`
  - `logistic_broad`: AURC `0.7568`, SelAcc@80% `22.50%`, SelAcc@50% `30.00%`
  - random / oracle references: AURC `0.8254` / `0.7196`
- Current read:
  - the Math500 reliability signal looks weaker than GSM8K/SVAMP but not random
  - `n_valid` and AWNF/Bucket-A-derived features seem to carry some ranking information
  - this is better read as a **weak positive** for the reliability track than as a strong abstention policy

- `2026-05-14`: an E4 abstention-as-retry simulation also completed from
  [reliability_retry_20260514.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/reliability_retry_20260514.md)
  and
  [reliability_retry_20260514.json](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/reliability_retry_20260514.json).
- Setup:
  - baseline answer is reconstructed `exp_only`
  - retry is an offline substitution on only the lowest-score samples, using already existing answer sources (`cgap_vote`, `prob_vote`, `blockactive_vote`, `final_answer`, `retry_majority`)
  - score variants tested: `combined_score` and `logistic_broad_score`
- Readout:
  - best GSM8K result comes from retrying into `prob_vote`
  - `combined_score` with a `20%` retry budget: `67.42% → 68.94%` (`+1.52pt`), with `4` fixes and `0` hurts on the test split
  - `logistic_broad_score` with a `25%` retry budget reaches the same `68.94%`
  - SVAMP transfer shows the same pattern more weakly: `86.33% → 86.67%` (`+0.33pt`)
- Current read:
  - this does not beat simply using `prob_vote` globally, but it recovers that gain with a much smaller retry budget
  - the reliability score therefore looks more useful for **targeted fallback** than for vote weighting

- `2026-05-14`: an E5 calibration pass also completed from
  [reliability_calibration_20260514.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/reliability_calibration_20260514.md)
  and
  [reliability_calibration_20260514.json](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/reliability_calibration_20260514.json).
- Readout:
  - GSM8K test: `ECE 0.0596`, `MCE 0.2778`, `Brier 0.1729`
  - SVAMP transfer: `ECE 0.1155`, `MCE 0.2853`, `Brier 0.1091`
  - Math500 test: `ECE 0.0442`, `MCE 0.0442`, `Brier 0.1646`
- Current read:
  - the learned GSM8K score is not perfectly calibrated, but it is reasonable enough to support threshold-based abstention / retry
  - SVAMP transfer looks somewhat underconfident in the mid/high-score bins, though the ranking signal still seems useful
  - Math500's low ECE is misleading on its own because the score mostly collapses into a narrow `0.2~0.3` band; this looks more like **weak resolution** than strong confidence calibration

- `2026-05-14`: an E6 threshold-based fallback policy also completed from
  [reliability_retry_policy_20260514.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/reliability_retry_policy_20260514.md)
  and
  [reliability_retry_policy_20260514.json](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/reliability_retry_policy_20260514.json).
- Policy:
  - if reliability score `< tau`, replace `exp_only` with `prob_vote`
  - score candidates tested: `combined_score`, `logistic_broad_score`
  - thresholds chosen on GSM8K validation only
- Best current policy:
  - `logistic_broad_score < 0.5845 -> prob_vote`
  - GSM8K test: `67.42% → 68.94%` (`+1.52pt`), with `4` fixes and `0` hurts
  - SVAMP: `86.33% → 86.67%` (`+0.33pt`), with `1` fix and `0` hurts
- Current read:
  - this is essentially the operationalized form of the E4 retry result
  - the learned score appears stable enough to support a simple rule-based fallback gate on the current artifact
  - the remaining gap to a “real” retry policy is not offline selection logic anymore, but fresh generation / latency / cost accounting

## Hybrid line closure

- The hybrid line bundles Phase 7 (sum), Phase 7B (exp-scale relaxation), Phase 7-filter (pruning), and Phase 8 offline proxy (single-event commit-equivalent).
- On the current GSM8K artifact, none of these four activation paths produced a clear test/transfer gain over `exp_only`, and a single proxy event already matches `exp_only` accuracy. We treat the hybrid line as **closed on the current artifact** rather than open-pending-Phase-5.5, because:
  - Phase 7-filter and Phase 8 proxy both directly test the gap signal without depending on patch quality.
  - Phase 2 left GSM8K AWNF at `0.2%`, so patch sensitivity on GSM8K is small a priori.
- Reopening conditions (if any later evidence calls for it): a substantively different signal family (for example, a true commit-time hook driven by an independent motivation, or a sample-level abstention/retry path that uses `max_gap` as a between-sample reliability score), or a structural change to the voting routine itself.

## Math500 Bucket A Status

- `2026-05-13`: parser-side Bucket A follow-up completed from
  [math500_bucket_a_20260513.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/math500_bucket_a_20260513.md)
  and
  [math500_bucket_a_20260513.json](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/math500_bucket_a_20260513.json).
- Readout:
  - `no_boxed_no_answer_tag`: `43.2%`
  - `boxed_like_but_unparseable`: `26.5%`
  - `boxedboxed_corruption`: `22.7%`
  - `no_boxed_but_answer_tag`: `7.5%`
- Current read:
  - Math500 parser-side AWNF seems to be a mix of missing recoverable answer spans and malformed boxed formatting, not one narrow parser bug
  - this keeps parser hardening / format-robustness analysis on the table and makes a pure char-offset repair look less likely to be the main fix by itself

- `2026-05-14`: an offline Bucket A rescue upper-bound pass also completed from
  [math500_bucket_a_rescue_20260514.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/math500_bucket_a_rescue_20260514.md)
  and
  [math500_bucket_a_rescue_20260514.json](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/math500_bucket_a_rescue_20260514.json).
- Readout:
  - `49.4%` of Bucket A events became parseable under conservative formatting-only heuristics
  - but only `3.9%` were GT-equivalent
  - sample-level `exp_only` upper bound moved only `24.40% → 24.80%` (`+0.40pt`), with `3` fixed samples and `1` hurt sample
- Current read:
  - formatting-only parser hardening can recover text shape, but most rescued strings are still not semantically useful answers
  - this makes parser-side hardening look more like a diagnostic / robustness branch than a likely first-order fix for the current Math500 ceiling

## Strict Anchor Debug Findings

- The strict anchor baseline changes the generation trajectory itself, not just the vote weights.
  - `constraints_text="96:The answer is"` is re-applied during generation, so every sample is forced to contain the phrase.
- The anchor phrase usually appears before the model naturally reaches the answer section.
  - `countdown`: 242/256 before `<answer>`, 219/256 before `</reasoning>`
  - `gsm8k`: 1157/1319 before `<answer>`, 1224/1319 before `</reasoning>`
  - `math500`: 439/500 before `<answer>`, 460/500 before `</reasoning>`
  - `svamp`: 247/300 before `<answer>`, 238/300 before `</reasoning>`
- This means the fixed anchor is acting like a forced early-answer prior, not a passive probe.
- The confidence window is fixed at suffix offset `100` for all valid events and is not tied to the parsed answer location.
  - In practice the strict weight often measures transition/formatting tokens near `The answer is`, `</reasoning>`, or `<answer>`, rather than the answer tokens themselves.
- `countdown` fails catastrophically because the task wants a valid expression using all provided numbers, but the forced phrase encourages direct verbal answers.
  - Example failures include:
    - target `18` with boxed `37 - 1` (missing `20`)
    - target `22` with boxed `96/7` (uses a number not in the prompt)
    - target `92` with boxed `38 + 36` (uses an intermediate value instead of the original numbers)
  - Final parse success drops from `146/256` in the earlier cgap run to `45/256` in strict.
- `gsm8k` mostly remains parseable, but it often commits to an answer too early and reasons incorrectly.
  - Final parse success improves (`1315/1319`), but final accuracy drops to `66.26%`, so the main failure is wrong reasoning, not parser collapse.
- `math500` keeps its final accuracy, but voting does not improve because intermediate answer labels remain noisy.
  - In strict `math500`, about `18.78%` of valid vote events have parsed-answer strings longer than `80` characters, showing that parser noise is still substantial.
- `svamp` is the least harmed because the tasks are short and simple, but strict still does not beat `exp`.

## Answer-Locate Prob-Gap Debug Findings

- `confidence_gap_answer_window5_prob_mean_rawsum` keeps generation unchanged and only changes the vote weight source from raw logit gap to probability gap.
- Probability gaps compress all per-step weights to `[0, 1]`, which reduces extreme outliers but also reduces score separation between competing answers.
- `countdown`
  - `13` vote answers changed relative to the earlier logit-gap run, but all `13` were wrong in both settings.
  - This suggests prob-gap mainly reshuffles low-confidence wrong candidates instead of rescuing the correct expression.
- `gsm8k`
  - `50` vote answers changed: `10` improved, `12` worsened, `28` changed among wrong answers only.
  - Several flips are caused by early high-probability short numeric candidates (for example `0`, `10`, `29`) accumulating enough bounded mass to outrank the later correct answer.
- `math500`
  - `58` vote answers changed: `7` improved, `3` worsened.
  - Even though `answer_window_not_found` remains very high (`33.62%` of events), the bounded prob-gap seems to help by preventing a few very large raw-logit events from dominating the vote.
- `svamp`
  - Only `4` vote answers changed, with `2` improvements and `1` regression.
  - This matches the final result: small but real gain over both the logit-gap locate variant and `exp`.

## Planned Work

> This tracker is a lightweight experiment roadmap, not the full implementation spec. Detailed phase mechanics live in [CGAP_IMPROVEMENT_PLAN.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/CGAP_IMPROVEMENT_PLAN.md).  
> Strict anchor experiments are **archived exploratory** — see Strict Anchor Debug Findings. Mainline focus is now the answer-locate prob variant with char-offset localization repair and possible hybrid follow-ups.

1. **[Phase 2]** Offline AWNF decomposition on Math500
   - Separate suspect parser outputs from likely true alignment failures
   - Use this as the go/no-go gate for the char-offset patch
2. **[Phase 3]** GSM8K gap-correctness signal check
   - Initial exploratory pass completed on `2026-05-12`
   - Re-run only if patched data changes the effective event population enough to revisit the signal estimate
3. **[Phase 4]** Char-offset localization patch for Math500 AWNF
   - `_locate_answer_window_v2()` using tokenizer `offset_mapping` instead of BPE subsequence search
   - Goal: check whether Math500 AWNF can be reduced meaningfully, especially on Bucket B cases
4. **[Phase 5/5.5]** Patched debug reruns — Math500 + GSM8K with `use_char_offset=True`
   - GSM8K rerun is conditional rather than automatic; if Phase 4 is not pursued or is unlikely to change GSM8K behavior materially, current GSM8K debug data may be treated as patched-equivalent for the hybrid line
5. **[Phase 6]** Post-patch reanalysis
   - Re-check AWNF reduction and compare post-patch signal quality, especially on Math500
6. **[Phase 7]** Hybrid / gating sweep on GSM8K patched data — **closed on current artifact**
   - Exploratory pre-patch passes completed on `2026-05-12` and `2026-05-13` (Phase 7, 7B, 7-filter)
   - No test/transfer gain over `exp_only` in any of the three formulations
   - Main pass deferred indefinitely — reopen only if a structurally new signal family motivates a fresh hybrid recipe
7. **[Phase 8]** Commit-time gap exploratory — **proxy completed; GPU hook deferred**
   - Offline `parsed_answer`-trajectory proxy completed on `2026-05-13`; see Phase 8 Status
   - Proxy upper-bounds the voting value of commit-gap; not recommended to instrument the real hook on the current artifact
   - Worth revisiting if a different application (sample-level abstention/retry using `max_gap` reliability) is taken up separately
8. **[Phase E]** Sample-level reliability for selective abstention — **first positive result** (`2026-05-13`)
  - Per-sample features (`max_gap`, `n_unique_answers`, `last_change_step_frac`, etc.) drive a risk-coverage policy that beats random and approaches oracle on GSM8K val/test and SVAMP transfer
  - Follow-up directions now center on how to use the score operationally; Math500 pass, offline retry, calibration, and a concrete threshold-policy gate all completed on `2026-05-14`

## Logging Workflow

Future runs can be appended automatically with:

```bash
cd /home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval
python log_experiment_results.py \
  outputs/LLaDA-8B-Instruct/<run_name> \
  --log-file /home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/CONFIDENCE_GAP_EXPERIMENTS.md \
  --section-title "<run_name>"
```

If `run_eval.sh` is used, set:

```bash
EXPERIMENT_LOG_FILE=/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/CONFIDENCE_GAP_EXPERIMENTS.md
EXPERIMENT_LOG_TITLE=<run_name>
```

and the run summary will be appended automatically.

## Automatic Run Log


## 2026-04-30 15:49:14 - 20260429_cgap_window5_blockactive_prob_mean_rawsum_bs4_all_debug

Source paths:
- `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260429_cgap_window5_blockactive_prob_mean_rawsum_bs4_all_debug`

Note: Auto-imported historical run

| Dataset | Batch | Gen | Steps | Vote Method | Final Acc | Vote Acc | Directory |
|---|---:|---:|---:|---|---:|---:|---|
| countdown | 4 | 128 | 64 | confidence_gap_answer_window5_blockactive_prob_mean_rawsum | 21.48 | 16.80 | /home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260429_cgap_window5_blockactive_prob_mean_rawsum_bs4_all_debug/countdown_gen128_steps64_vote_confidence_gap_answer_window5_blockactive_prob_mean_rawsum_bs4 |
| gsm8k | 4 | 128 | 64 | confidence_gap_answer_window5_blockactive_prob_mean_rawsum | 68.69 | 69.07 | /home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260429_cgap_window5_blockactive_prob_mean_rawsum_bs4_all_debug/gsm8k_gen128_steps64_vote_confidence_gap_answer_window5_blockactive_prob_mean_rawsum_bs4 |
| math | 4 | 128 | 64 | confidence_gap_answer_window5_blockactive_prob_mean_rawsum | 27.00 | 23.80 | /home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260429_cgap_window5_blockactive_prob_mean_rawsum_bs4_all_debug/math500_gen128_steps64_vote_confidence_gap_answer_window5_blockactive_prob_mean_rawsum_bs4 |
| svamp | 4 | 128 | 64 | confidence_gap_answer_window5_blockactive_prob_mean_rawsum | 84.67 | 84.67 | /home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260429_cgap_window5_blockactive_prob_mean_rawsum_bs4_all_debug/svamp_gen128_steps64_vote_confidence_gap_answer_window5_blockactive_prob_mean_rawsum_bs4 |


## 2026-04-30 19:10:33 - 20260430_cgap_anchor_window5_logit_mean_rawsum_bs4_all_debug

Source paths:
- `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260430_cgap_anchor_window5_logit_mean_rawsum_bs4_all_debug`

| Dataset | Batch | Gen | Steps | Vote Method | Final Acc | Vote Acc | Directory |
|---|---:|---:|---:|---|---:|---:|---|
| countdown | 4 | 128 | 64 | confidence_gap_anchor_window5_logit_mean_rawsum | 4.69 | 5.08 | /home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260430_cgap_anchor_window5_logit_mean_rawsum_bs4_all_debug/countdown_gen128_steps64_vote_confidence_gap_anchor_window5_logit_mean_rawsum_bs4 |
| gsm8k | 4 | 128 | 64 | confidence_gap_anchor_window5_logit_mean_rawsum | 66.26 | 67.10 | /home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260430_cgap_anchor_window5_logit_mean_rawsum_bs4_all_debug/gsm8k_gen128_steps64_vote_confidence_gap_anchor_window5_logit_mean_rawsum_bs4 |
| math | 4 | 128 | 64 | confidence_gap_anchor_window5_logit_mean_rawsum | 27.00 | 27.00 | /home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260430_cgap_anchor_window5_logit_mean_rawsum_bs4_all_debug/math500_gen128_steps64_vote_confidence_gap_anchor_window5_logit_mean_rawsum_bs4 |
| svamp | 4 | 128 | 64 | confidence_gap_anchor_window5_logit_mean_rawsum | 85.00 | 86.33 | /home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260430_cgap_anchor_window5_logit_mean_rawsum_bs4_all_debug/svamp_gen128_steps64_vote_confidence_gap_anchor_window5_logit_mean_rawsum_bs4 |


## 2026-04-30 23:43:52 - 20260430_cgap_answer_window5_prob_mean_rawsum_bs4_all_debug

Source paths:
- `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260430_cgap_answer_window5_prob_mean_rawsum_bs4_all_debug`

| Dataset | Batch | Gen | Steps | Vote Method | Final Acc | Vote Acc | Directory |
|---|---:|---:|---:|---|---:|---:|---|
| countdown | 4 | 128 | 64 | confidence_gap_answer_window5_prob_mean_rawsum | 21.48 | 23.05 | /home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260430_cgap_answer_window5_prob_mean_rawsum_bs4_all_debug/countdown_gen128_steps64_vote_confidence_gap_answer_window5_prob_mean_rawsum_bs4 |
| gsm8k | 4 | 128 | 64 | confidence_gap_answer_window5_prob_mean_rawsum | 68.69 | 69.67 | /home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260430_cgap_answer_window5_prob_mean_rawsum_bs4_all_debug/gsm8k_gen128_steps64_vote_confidence_gap_answer_window5_prob_mean_rawsum_bs4 |
| math | 4 | 128 | 64 | confidence_gap_answer_window5_prob_mean_rawsum | 27.00 | 25.60 | /home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260430_cgap_answer_window5_prob_mean_rawsum_bs4_all_debug/math500_gen128_steps64_vote_confidence_gap_answer_window5_prob_mean_rawsum_bs4 |
| svamp | 4 | 128 | 64 | confidence_gap_answer_window5_prob_mean_rawsum | 84.67 | 86.67 | /home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260430_cgap_answer_window5_prob_mean_rawsum_bs4_all_debug/svamp_gen128_steps64_vote_confidence_gap_answer_window5_prob_mean_rawsum_bs4 |
