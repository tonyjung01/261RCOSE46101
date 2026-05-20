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

- **Update (`2026-05-16`) — Layer 4 token ordering becomes the first clean raw-accuracy lift in the project**:
  - this line does **not** change the answer-level vote; it keeps `VOTE_METHOD=exp` fixed and changes only the decoder-side transfer ranking inside the active block
  - baseline transfer score: `p_top1` (`top1_prob`)
  - ablation transfer score: `p_top1 - p_top2` (`prob_margin`)
  - full-run `vote_answer` gains under the same deterministic `T=0` setup:
    - GSM8K: `69.67% -> 70.81%` (`+1.14pt`)
    - SVAMP: `86.00% -> 87.33%` (`+1.33pt`)
    - MATH500: `27.60% -> 27.60%` (`+0.00pt`)
    - Countdown: `23.05% -> 23.05%` (`+0.00pt`)
  - `final_answer` also improves on the parser-clean tasks:
    - GSM8K: `68.39% -> 69.37%` (`+0.98pt`)
    - SVAMP: `84.33% -> 86.67%` (`+2.34pt`)
  - read: answer-level voting ideas look saturated, but **token-level decoding order still has real headroom**
  - this should be framed as a decoder-policy ablation, not as a new vote-weight method
  - companion docs: `TOKEN_ORDERING_EXPERIMENT.md`, `eval/analysis/transfer_score_ablation_full_20260516.md`
  - **Update (`2026-05-20`) — first gated temporal redesign is mixed, not dominant**:
    - compared runs: `prob_margin` vs `gated_temporal_margin = margin + λ · stability · 1[margin < τ]`
    - fixed setup remained the same: `T=0`, same parser/prompt/model, same `exp` vote, same seed, same `bs=4`
    - full-run vote deltas vs `prob_margin`:
      - GSM8K: `70.81% -> 70.58%` (`-0.23pt`)
      - SVAMP: `87.33% -> 88.67%` (`+1.34pt`)
      - MATH500: `27.60% -> 28.40%` (`+0.80pt`)
      - Countdown: `23.05% -> 23.44%` (`+0.39pt`)
    - read: the temporal term is not universally bad, but the first gated redesign still does **not** beat `prob_margin` on the main reference task (`GSM8K`)
    - companion doc: `eval/analysis/gated_temporal_margin_full_20260520.md`

### Layer 4 quick reference — what the baselines and follow-ups actually are

#### Baseline metrics

| Name | What it means | Where it reads from | Why it matters |
|---|---|---|---|
| `final_answer` | parse the **final decoded string once** | only the last generation state | simplest baseline; no trajectory aggregation |
| `exp voting` / `vote_answer` | parse the answer at many diffusion steps and aggregate with exponentially increasing late-step weights | full stepwise answer trajectory | canonical `dLLM-MidTruth` baseline and the primary metric for Layer 4 |

**Important**: the token-ordering line does **not** replace `exp` voting. It keeps `VOTE_METHOD=exp` fixed and changes only the decoder-side ordering of which masked token positions are opened first.

#### Token-ordering experiment ladder

| Label | Transfer score | What changed | Scope | Current read |
|---|---|---|---|---|
| `A` | `top1_prob = p_top1` | baseline token-order score | full | decoder baseline for Layer 4 |
| `B` | `prob_margin = p_top1 - p_top2` | Kim-style margin ordering | full | first clean raw-accuracy lift |
| `C1` | `temporal_margin = margin + λ·runlen` | naive temporal extension | GSM8K smoke only | negative |
| `C-next-1` | `gated_temporal_margin = margin + λ·stability·1[margin < τ]` | gated temporal extension | full | mixed tradeoff, not dominant |

#### Full A/B result (`top1_prob` -> `prob_margin`)

| Task | `A final` | `A vote` | `B final` | `B vote` | Vote delta |
|---|---:|---:|---:|---:|---:|
| GSM8K | 68.39 | 69.67 | 69.37 | 70.81 | `+1.14pt` |
| SVAMP | 84.33 | 86.00 | 86.67 | 87.33 | `+1.33pt` |
| MATH500 | 27.00 | 27.60 | 27.20 | 27.60 | `+0.00pt` |
| Countdown | 19.53 | 23.05 | 18.36 | 23.05 | `+0.00pt` |

#### GSM8K smoke result for `C1` (`n=64`)

| Condition | Vote | Final |
|---|---:|---:|
| `A = top1_prob` | 76.56 | 76.56 |
| `B = prob_margin` | **79.69** | **78.12** |
| `C1a = temporal_margin, λ=0.05` | 75.00 | 73.44 |
| `C1b = temporal_margin, λ=0.10` | 76.56 | 73.44 |
| `C1c = temporal_margin, λ=0.20` | 75.00 | 71.88 |

#### Full `C-next-1` result (`prob_margin` -> `gated_temporal_margin`)

| Task | `prob_margin final` | `prob_margin vote` | `gated final` | `gated vote` | Vote delta |
|---|---:|---:|---:|---:|---:|
| GSM8K | 69.37 | 70.81 | 68.84 | 70.58 | `-0.23pt` |
| SVAMP | 86.67 | 87.33 | 88.00 | 88.67 | `+1.34pt` |
| MATH500 | 27.20 | 27.60 | 28.00 | 28.40 | `+0.80pt` |
| Countdown | 18.36 | 23.05 | 19.92 | 23.44 | `+0.39pt` |

#### Parser behavior by task

| Task | Parser function | Main extraction rule | Correctness check | Practical implication |
|---|---|---|---|---|
| GSM8K | `parse_gsm_answer()` | first valid `\boxed{...}` numeric answer; fallback to `<answer>...</answer>` numeric extraction | exact float equality | relatively parser-clean |
| SVAMP | `parse_svamp_answer()` | same basic numeric boxed-answer logic as GSM8K | exact float equality | also parser-clean enough for decoder gains to show up |
| MATH500 | `parse_math_answer()` | take the **last** boxed answer string; fallback to `<answer>...</answer>` raw text | symbolic/string equivalence via `is_equiv()` | parser/format bottleneck is much stronger |
| Countdown | `parse_ctd_answer()` | extract boxed or answer-tag expression, normalize operators, require valid expression structure | validate numbers used and evaluate expression to target | strongest structural parser/evaluator bottleneck |

This parser split is important for interpreting Layer 4:

- `GSM8K` / `SVAMP`: decoder-policy gains tend to show up directly in accuracy
- `MATH500` / `Countdown`: trajectory changes may be real, but parser / evaluator bottlenecks can hide or dampen the gain

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
- The clearest current operational direction is now in the reliability track:
  - a simple offline fallback rule,
    `if logistic_broad_score < 0.5845 -> prob_vote else keep exp_only`,
    still gives a useful low-cost reference point: GSM8K `67.42% → 68.94%` (`+1.52pt`) and SVAMP `86.33% → 86.67%` (`+0.33pt`)
  - **archive note (`2026-05-15`)**: all later `T=0` rerun-based "retry" analyses are now treated as exploratory only, not as load-bearing evidence
    - under a truly identical `T=0` setup, rerunning should be deterministic
    - our rerun artifacts were not fully identical to the base artifact (e.g. stored batch size / model path / run metadata differ, and older base artifacts did not persist enough metadata to certify end-to-end identicality)
    - so apparent `T=0` rerun gains are kept as artifact probes, not as claims about a meaningful retry mechanism
  - a real `T>0` retry-pool pilot is still kept as a **separate self-consistency probe**, not as direct extra support for the reliability score:
    - `T=0.5` collapsed at K=1, so the viable run used `T=0.2`, `K=3`, seeds `42/43/44`
    - answer diversity is real (`49/60` and `45/60` answer diffs vs seed42 for seeds 43/44)
    - but deployable gain is weak: `majority(K)` full-test acc `70.08% / 69.70% / 70.45%` for `K=1/2/3`
    - merged step-level re-vote is worse: `70.08% / 68.56% / 68.56%`
    - confidence-based in-pool selectors also fail to beat majority (`best_by_margin 68.56%`, `best_by_top1 68.56%`, `confidence_weighted 68.94%`)
    - read: regeneration diversity exists, but current aggregation rules capture little of it; this is a separate self-consistency-side negative mechanism finding
  - rerun-based cross-task / budget-scaling analyses remain on disk, but should now be read with the same archive note as the GSM8K `T=0` rerun probes
- **Update (2026-05-14)** — a differential targetability followup
  ([reliability_differential_20260514.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/reliability_differential_20260514.md),
  [reliability_differential_20260514.json](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/reliability_differential_20260514.json),
  [reliability_differential_20260514_interpretation.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/reliability_differential_20260514_interpretation.md))
  reframes that rule:
  - `Pearson(logistic_broad_score, delta=y_prob−y_exp) ≈ 0` on val (−0.003) and weak on test (−0.129); `Pearson(score, y_exp) ≈ +0.45` — so the score predicts **difficulty**, not **method-switch utility**
  - on the seed=42 test split, global `prob_vote` was `70.08%` (`+2.66pt`), while the gated rule was `68.94%` (`+1.52pt`) — single-seed reading suggested the gating was dominated
- **Further update (2026-05-14)** — a multi-seed robustness pass
  ([reliability_multiseed_20260514.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/reliability_multiseed_20260514.md),
  [reliability_multiseed_20260514.json](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/reliability_multiseed_20260514.json),
  [reliability_multiseed_20260514_interpretation.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/reliability_multiseed_20260514_interpretation.md))
  partially reverses the single-seed differential conclusion:
  - across 10 outer seeds, mean test acc — `exp_only 69.05% ± 1.96pt`, `prob_vote 68.94% ± 1.93pt`, `gated 69.66% ± 1.82pt`
  - `gated − exp_only`: `+0.61pt ± 0.78pt` (gated beats `exp_only` on 70% of seeds)
  - `gated − prob_vote`: `+0.72pt ± 1.11pt` (gated beats global `prob_vote` on 70% of seeds)
  - `prob_vote − exp_only`: `−0.11pt ± 1.55pt` (global swap is neutral on average; seed=42's `+2.66pt` was a lucky single-seed outcome with `exp_better=0`)
  - reconciliation: the gating works through **budget-bounded swap + weak targeting**, not through strong targeting. `Pearson(score, delta) ≈ 0` is still true, but limiting swap to the bottom 25% by score is enough to skew fix:hurt ratio favorably
- Net read after both followups:
  - the E6 threshold rule has a real but smaller effect than first reported (`+0.61pt` mean vs `+1.52pt` single-seed), and it does dominate the global `prob_vote` swap on average
  - the load-bearing reliability finding remains the Phase E abstention AURC; the threshold-rule swap is a secondary operational artifact
  - all later `T=0` rerun-based "retry" / control / decomposition passes are now archived as artifact probes only
  - those files remain useful for auditing how the non-identical rerun artifacts behaved, but they are no longer promoted as support for a meaningful retry mechanism
  - **Update (`2026-05-15`) — same-artifact router / coalition probes are also null**:
    - `router_oracle_phase1_20260515.{md,json}` shows that val-selected single-readout deployment still chooses `exp_only`, so test gain is `+0.00pt`
    - `coalition_override_phase2b1_20260515.{md,json}` shows that the val-best conservative coalition rule is effectively “do nothing” (`0` overrides on test, `+0.00pt`)
    - the full oracle ceiling is still `70.83%` (`+3.41pt`), but that headroom lives inside a small disagreement slice (`33/264`) and is not captured by simple val-stable routing rules
    - read: under the same `T=0` artifact, raw-accuracy improvement is saturated not only for vote weighting, but also for single-readout replacement and conservative within-artifact routing

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
  - this is essentially the operationalized form of the E4 offline fallback result
  - the learned score appears stable enough to support a simple rule-based fallback gate on the current artifact
  - this should be read as a fallback-gating result, not as evidence for regeneration-based retry
- **Update (2026-05-14)** — the differential targetability followup
  ([reliability_differential_20260514.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/reliability_differential_20260514.md))
  shows this rule's lift is mostly driven by global `prob_vote` advantage on the gsm8k test split (which has `exp_better=0`, `prob_better=7`), not by score targeting:
  - `Pearson(score, delta=y_prob−y_exp) ≈ 0` while `Pearson(score, y_exp) ≈ +0.45` — score predicts difficulty, not method-switch utility
  - global `prob_vote` on the same test split is `70.08%` (`+2.66pt`), strictly better than the gated `68.94%`
  - on val (the fit split) score-gated swap is not better than random gating at budgets ≥ 25%
  - **the threshold rule should not be promoted as "operational best" until a multi-seed pass confirms robustness**

- **Archive note (`2026-05-15`)**: the sections below retain the rerun results for traceability, but they are no longer load-bearing. Under truly identical `T=0` settings a rerun should be deterministic; these comparisons therefore reflect non-identical artifact conditions, not a clean retry mechanism.

- `2026-05-14`: an archived selective-rerun artifact comparison also completed from
  [true_retry_eval_gsm8k_exp_test_tau05845_20260514_v6.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/true_retry_eval_gsm8k_exp_test_tau05845_20260514_v6.md),
  [true_retry_eval_gsm8k_exp_test_tau05845_20260514_v6.json](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/true_retry_eval_gsm8k_exp_test_tau05845_20260514_v6.json),
  [true_retry_eval_gsm8k_prob_test_tau05845_20260514_v1.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/true_retry_eval_gsm8k_prob_test_tau05845_20260514_v1.md),
  and
  [true_retry_eval_gsm8k_prob_test_tau05845_20260514_v1.json](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/true_retry_eval_gsm8k_prob_test_tau05845_20260514_v1.json).
- Setup:
  - same reliability score and threshold as E6: `logistic_broad_score < 0.5845`
  - retry only the bottom `60 / 264` GSM8K test samples (`22.73%`)
  - compare retrying into the rerun artifact's `exp_only` answer versus its voted answer
- Readout:
  - baseline `exp_only`: `67.42%`
  - E6 offline fallback: `68.94%` (`+1.52pt`), `4` fixes, `0` hurts
  - archived rerun comparison into rerun `exp_only`: `70.08%` (`+2.65pt`), `8` fixes, `1` hurt
  - archived rerun comparison into rerun voted answer: also `70.08%` (`+2.65pt`), `8` fixes, `1` hurt
- Current read:
  - these numbers are now treated as an artifact-level rerun comparison only
  - because the `T=0` rerun was not certified as a fully identical deterministic rerun, this section should not be promoted as method evidence
  - the useful surviving lesson is only that the **score identified a hard slice**; the rerun gain itself is archived

## Phase E-Retry-Control Status (archived exploratory T=0 rerun control)

> Archive note: this section remains useful as a record of how the rerun artifacts behaved, but it is no longer promoted as direct evidence. The underlying comparisons depend on `T=0` rerun artifacts that were not certified as fully identical deterministic reruns.

- `2026-05-14`: pure random control completed from
  [retry_control_eval_20260514.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/retry_control_eval_20260514.md),
  [retry_control_eval_20260514.json](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/retry_control_eval_20260514.json),
  and random-subset manifest
  [random_retry_subset_test_seed123_20260514.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/random_retry_subset_test_seed123_20260514.md).
- Setup:
  - same retry budget as the selective run (`60/264 = 22.73%` of gsm8k test)
  - random subset drawn from full test pool with `seed=123` (pure random, not complement-only); 12 of 60 happen to overlap with the score-flagged set
  - same retry pipeline (`run_retry_policy_experiment.sh`), `vote_method=exp`, retry artifact at `outputs/.../20260514_gsm8k_retry_exp_random60_seed123/`
  - `--answer-kind exp_only` on the retry (matches the selective evaluation)
- Readout (single seed, gsm8k test n=264, current artifact):
  - P0 baseline `exp_only`: `67.42%`
  - P_selective (score-targeted retry): `70.08%` (`+2.65pt`), `8` fixes, `1` hurt, `28` changed
  - P_random (random-budget retry): `67.80%` (`+0.38pt`), `4` fixes, `3` hurts, `13` changed
  - **targeting marginal value (P_selective − P_random) = `+2.27pt`** on this artifact
- Subset-local (only the retried samples):
  - selective subset: base `30.00%`, retry `41.67%`, delta `+11.67pt`
  - random subset:    base `68.33%`, retry `70.00%`, delta `+1.67pt`
  - the substantially larger retry delta on the selective slice (`+11.67pt` vs `+1.67pt`) is the cleanest direct evidence on this artifact that the reliability score is concentrating retry budget on samples with real retry headroom; same GPU mechanism, different sample selection. Magnitudes on n=60 with a single random draw should be read as artifact-level numbers, not as a fixed ratio.
- Current read (artifact-level):
  - on this artifact, score-targeted retry beats random-budget retry by `+2.27pt` full-test and by a substantially larger subset-local delta; the selective-retry result is not just "retry helps in general"
  - random-budget retry has a small positive bias (`+0.38pt` full-test, `4` fixes vs `3` hurts) — retry itself is close to zero-sum without targeting
  - decomposing P_selective's `+2.65pt`: about `+0.38pt` is "retry helps in general", about `+2.27pt` is the score-targeting marginal — on this artifact, single seed
- Caveats:
  - single outer seed, single retry artifact per policy (selective `v6`, random `seed=123`); magnitudes are noisy and the sign itself is provisional
  - `12/60` overlap between random and selective subsets gives the random control a small upward bias from including some flagged samples — the complement-only control below addresses this
  - the targeting marginal value is measured against `exp_only` retry artifacts; the `prob` retry path (`v1`) lands at the same `25/60` subset-local accuracy on the selective slice, but a parallel random control with `prob` voting has not been run

- `2026-05-14`: a complement-only random control completed from
  [retry_control_complement_eval_20260514.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/retry_control_complement_eval_20260514.md),
  [retry_control_complement_eval_20260514.json](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/retry_control_complement_eval_20260514.json),
  and interpretation notes
  [retry_control_complement_eval_20260514_interpretation.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/retry_control_complement_eval_20260514_interpretation.md).
- Setup:
  - same retry budget (`60` samples) as the selective and pure-random controls
  - random subset drawn from the unflagged 204 only (`seed=124`, `--exclude-flagged`), giving `0/60` overlap with the score-flagged set
  - retry artifact: `outputs/.../20260514_gsm8k_retry_exp_complement_seed124/`
- Readout (single seed, gsm8k test n=264, current artifact):
  - P0 baseline: `67.42%`
  - P_selective: `70.08%` (`+2.65pt`), `8` fixes, `1` hurt
  - P_random_complement: `67.05%` (`−0.38pt`), `2` fixes, `3` hurts, `6` changed
  - **targeting marginal under complement control: `+3.03pt`** (vs `+2.27pt` under pure-random seed=123)
- Subset-local on the retried samples:
  - selective subset: `30.00% → 41.67%` (`+11.67pt`)
  - complement random subset: `76.67% → 75.00%` (`−1.67pt`) — i.e. retry on unflagged samples is **net-negative** on this artifact
- Clean targeting gradient across the three controls on this artifact:
  - retry on score-flagged 60 (selective): subset-local delta `+11.67pt`
  - retry on pure-random 60 (12/60 overlap with flagged): subset-local delta `+1.67pt`
  - retry on complement 60 (0/60 overlap with flagged): subset-local delta `−1.67pt`
  - monotone in "fraction of retry budget spent on flagged samples"; consistent with the score concentrating retry budget on samples that actually benefit
- Current read:
  - the stricter overlap-free control gives a slightly larger targeting marginal (`+3.03pt`) than the pure-random control (`+2.27pt`); the pure-random control was slightly deflating the targeting effect via the 12 flagged samples it leaked in
  - "retry on unflagged samples is net-negative" is single-seed at this point and gets refined under multi-seed below
  - the multi-seed refinement (4 complement seeds) is in the next entry

- `2026-05-15`: multi-seed random-control pass (random-control-seed only, single outer split) completed from
  [retry_control_multiseed_eval_20260515.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/retry_control_multiseed_eval_20260515.md),
  [retry_control_multiseed_eval_20260515.json](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/retry_control_multiseed_eval_20260515.json),
  and interpretation notes
  [retry_control_multiseed_eval_20260515_interpretation.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/retry_control_multiseed_eval_20260515_interpretation.md). *(Naming note: "multi-seed" here refers to varying the random-control subset seed only. The outer val/test split (`seed=42`), the score fit on GSM val, and the selective subset are all held fixed. This is **not** a method-level multi-seed; outer-split robustness is a separate, deferred experiment.)*
- Setup:
  - archived rerun policy, score, threshold, outer split, answer kind held fixed; only the random-control draw varies
  - `K=4` complement-only random controls (seeds `124, 125, 126, 127`, all `--exclude-flagged` → `0/60` overlap with the flagged set)
  - retry artifacts: `outputs/.../20260514_gsm8k_retry_exp_complement_seed{124,125,126,127}/`
- Per-seed targeting marginals (`P_selective − P_random`):
  - seed=124: `+3.03pt`
  - seed=125: `+3.79pt`
  - seed=126: `+2.27pt`
  - seed=127: `+3.03pt`
- Aggregate:
  - **targeting marginal: mean `+3.03pt` ± `0.62pt` across 4 seeds** (approx 95% CI on the mean `≈ [+2.42pt, +3.64pt]`, cleanly excludes 0)
  - all `4/4` seeds give positive targeting marginal
  - random subset-local retry delta: mean `−1.67pt ± 2.72pt` (wide); `3/4` seeds have negative delta, seed=126 shows `+1.67pt`
  - magnitude range across seeds: `[+2.27pt, +3.79pt]` — `~4` samples of spread on n=264
- Current read:
  - the targeting marginal is robust to **random-control draw on the current outer split**; it is not a one-shot random-subset artifact
  - the pure-random `+2.27pt` (seed=123, 12/60 overlap) sits at the low end of this range, consistent with the 12 flagged samples slightly deflating it
  - "retry on unflagged is net-negative" is a directional finding (3/4 seeds, wide std), not a strict claim
  - **scope reminder**: this multi-seed sweep varies the random-control subset only; the outer split, the score fit, and the selective subset are all held fixed. Method-level robustness across new val/test splits (re-fit score, re-flag) is a separate question and deferred to the robustness/writeup stage

## Phase E-Retry-CrossTask Status (SVAMP, archived rerun portability probe) — mixed transfer

- `2026-05-15`: SVAMP cross-task retry control completed from
  [svamp_retry_control_eval_20260515.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/svamp_retry_control_eval_20260515.md),
  [svamp_retry_control_eval_20260515.json](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/svamp_retry_control_eval_20260515.json),
  and interpretation notes
  [svamp_retry_control_eval_20260515_interpretation.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/svamp_retry_control_eval_20260515_interpretation.md).
- Setup:
  - GSM8K-val-fitted `logistic_broad` score + `tau=0.5845` applied to SVAMP without any SVAMP-specific tuning
  - archived rerun policy on SVAMP samples with score `≤ tau` (`49/300 = 16.33%`, flagged base acc `55.10%`)
  - random complement control (seed=124, `--exclude-flagged`): 49 random samples from the unflagged 251 (overlap with flagged = 0, base acc `93.88%`)
  - retry artifacts: `outputs/.../20260515_svamp_retry_exp_selective_tau05845/`, `outputs/.../20260515_svamp_retry_exp_complement_seed124/`
- Readout (single seed, SVAMP n=300):
  - P0 baseline: `86.33%`
  - P_selective: **`86.33%` (`+0.00pt`)** — `3` fixes, `3` hurts on the flagged 49
  - P_random complement: `85.67%` (`−0.67pt`) — `0` fixes, `2` hurts
  - targeting marginal `(P_selective − P_random) = +0.67pt` on this artifact (≈ `2` samples on n=300)
- Subset-local (retried slice):
  - selective subset: `55.10% → 55.10%` (`+0.00pt`)
  - random complement subset: `93.88% → 89.80%` (`−4.08pt`)
- Cross-task comparison:
  - GSM8K (seed=42, multi-seed K=4 random): selective subset retry delta `+11.67pt`, targeting marginal `+3.03pt ± 0.62pt`
  - SVAMP (single seed): selective subset retry delta `+0.00pt`, targeting marginal `+0.67pt`
- Current read:
  - what transfers: "retry on confident samples is net-negative" — SVAMP complement subset retry delta is `−4.08pt`, sharper than GSM8K's `−1.67pt` (consistent with SVAMP's higher base on the unflagged slice)
  - what does not transfer: "flagged rerun samples produce a lift" — collapses to `+0.00pt` on SVAMP. Same score, same threshold, same rerun pipeline; the lift mechanism does not survive cross-task
  - plausible explanation, consistent with the GSM8K Phase E-Diff finding: the score predicts *difficulty*, not *archived rerun utility*. On GSM8K those two correlated; on SVAMP they decorrelate — flagged SVAMP samples are difficult, but not the kind that happened to improve under this rerun artifact family
  - the `+0.67pt` targeting marginal on SVAMP is mostly driven by the random control hurting, not by selective helping; on `n=300` that is `~2` samples, at the boundary of signal vs noise
- Caveats:
  - single SVAMP retry per policy; no multi-seed yet for the SVAMP control
  - `3 fixes / 3 hurts` exact balance on the selective subset (n=49) is small-sample
  - this pass intentionally reuses the GSM8K-val-fitted logistic_broad; a SVAMP-tuned follow-up is reported below
- Operational summary so far:
  - GSM8K archived rerun rule (`tau=0.5845`, rerun bottom 22.73%) — produced a lift on the current artifact, but under the `T=0` caveat this remains an archived observation rather than a main claim
  - **same rule does not produce a SVAMP lift** on the current single-seed SVAMP artifact
  - the "don't retry confident samples" half of the rule transfers cleanly; the "retry flagged samples for gain" half does not

- `2026-05-15`: **SVAMP-tuned retry control** completed from
  [svamp_tuned_retry_control_eval_20260515.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/svamp_tuned_retry_control_eval_20260515.md),
  [svamp_tuned_retry_control_eval_20260515.json](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/svamp_tuned_retry_control_eval_20260515.json),
  and interpretation notes
  [svamp_tuned_retry_control_eval_20260515_interpretation.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/svamp_tuned_retry_control_eval_20260515_interpretation.md).
  This is the mechanism follow-up to the cross-task null: refit `logistic_broad` on SVAMP val (80/20 outer, seed=42), apply to SVAMP test, retry the flagged subset.
- Setup:
  - SVAMP val n=240, SVAMP test n=60 (small)
  - SVAMP-tuned tau at val 25%-quantile (`tau=0.8697`, higher than GSM8K's `0.5845` because the score distribution is more compressed on SVAMP)
  - flagged on test: 14/60 (base 57.14%); unflagged 46/60 (base 95.65%)
  - SVAMP-tuned flagged overlaps with GSM8K-transferred flagged on `7/8` of GSM8K-flagged test samples; SVAMP-tuned adds `7` and drops `1`
  - offline diagnostic: `prob_vote can_fix / base_wrong = 0/6` (no static fallback has the correct answer for any base-wrong flagged sample)
- Readout (single seed, SVAMP test n=60):
  - P0 baseline: `86.67%`
  - P_selective: `88.33%` (`+1.67pt`), `1` fix, `0` hurts on the flagged 14
  - P_random complement (seed=124): `86.67%` (`+0.00pt`), `0` fixes, `0` hurts (random subset had base 100% on this seed; T=0 deterministic kept all answers)
  - targeting marginal `+1.67pt`; selective subset-local retry delta `+7.14pt` (=`1` more correct out of `14`)
- Archived per-base-wrong rerun counts across tasks:
  - GSM8K (selective v6, flagged 60, base-wrong 42): rerun artifact flips `8/42 = ~19%`
  - SVAMP-tuned (this pass, flagged 14, base-wrong 6): rerun artifact flips `1/6 = ~17%`
  - SVAMP-transferred (prior cross-task, flagged 49, base-wrong 22): rerun artifact flips `3/22 = ~14%` (with `3` hurts)
  - these numbers are preserved as artifact-level context only; they should not be promoted as evidence for a portable retry mechanism
- Mechanism reading (synthesis with prior cross-task pass):
  - the earlier SVAMP cross-task `+0.00pt` looks less like "SVAMP retry is fundamentally broken" and more like a budget × score-precision wash: GSM8K-transferred score picked a slightly off SVAMP flagged set, and the 49-sample budget exposed enough hurt opportunities to balance the `3` fixes
  - SVAMP-tuned score on SVAMP test gives the cleaner archived rerun version: small flagged budget, no hurts, `1` flip out of `6` base-wrong samples
  - the 1 SVAMP rerun fix is unique within the archived rerun comparison (no offline source had the answer), but this stays inside the same archive-only caveat
- Caveats:
  - SVAMP test `n=60`, flagged budget `14`; a single-sample shift moves the headline
  - single seed, single retry per policy; random complement happened to be all-correct so the random side is a strict downside-only test on this run
  - n_val=240 is small for fitting 9 features; the SVAMP-tuned score is fit on this val and applied to the same outer split's test (no test leakage, but variance is wide)

- `2026-05-15`: **SVAMP-tuned at larger budget (q40)** completed from
  [svamp_tuned_q40_retry_control_eval_20260515.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/svamp_tuned_q40_retry_control_eval_20260515.md),
  [svamp_tuned_q40_retry_control_eval_20260515.json](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/svamp_tuned_q40_retry_control_eval_20260515.json),
  and interpretation notes
  [svamp_tuned_q40_retry_control_eval_20260515_interpretation.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/svamp_tuned_q40_retry_control_eval_20260515_interpretation.md).
- Setup: same SVAMP val/test split, same SVAMP-tuned score; tau widened to val 40%-quantile (`tau=0.9192`, vs q25 `0.8697`). Test flagged budget grew 14 → 23. Pre-GPU offline diagnostic: `prob_vote can_fix / base_wrong = 1/8` (q25 was 0/6).
- Readout (single seed, SVAMP test n=60):
  - P_selective: `88.33%` (`+1.67pt`), **`2` fixes, `1` hurt** on flagged 23
  - P_random complement (seed=124): `86.67%` (`+0.00pt`), 0/0 (unflagged base 100%, T=0 deterministic)
  - targeting marginal `+1.67pt` (same as q25 — no headline change at wider budget)
- Q25 vs Q40 decomposition:
  - q25 → q40: budget grew by 9 samples; those 9 contained 2 more base-wrong and 7 more base-correct
  - marginal contribution from those 9: `+1` fix, `+1` hurt → net `0`
  - per-flagged net rescue: q25 `1/14 ≈ 7.1%` → q40 `1/23 ≈ 4.3%` — wider budget is less efficient per retried sample
  - per-base-wrong rescue rate looks higher at q40 (`2/8 = 25%` vs q25 `1/6 = 17%`), but that hides the hurt that emerged on the borderline-confident slice
- Mechanism reading:
  - "rescue rate scales linearly with budget" hypothesis is **rejected on this artifact**; instead, the observed rerun gains concentrate at the very lowest-score tail and the marginal samples are a mix of "still flippable here" and "borderline-confident and flippable"
  - best operational budget on SVAMP-tuned is **the tightest (q25)**; wider tau gives the same net accuracy with more hurts
  - the q40 marginal rescue happens to be the offline-equivalent fix (`prob_vote` could have rescued it), so the uniquely retry-rescued count did not grow at the wider budget — only the offline-rescuable share added at the margin
- Caveats:
  - SVAMP test small (n=60); each fix/hurt is one sample; "marginal samples = net-zero" is structural but the exact `+1 / −1` count is noise-prone
  - random complement is structurally near-no-op on SVAMP at T=0 (unflagged is saturated and deterministic); the targeting marginal `+1.67pt` is therefore "selective made +1 fix while random did nothing", not a head-to-head signal
  - single seed

## Phase E-Retry-BudgetSweep Status (archived T=0 rerun budget probe)

- `2026-05-15`: GSM8K T=0 retry budget sweep completed from
  [gsm8k_retry_budget_sweep_20260515.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/gsm8k_retry_budget_sweep_20260515.md),
  [gsm8k_retry_budget_sweep_20260515.json](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/gsm8k_retry_budget_sweep_20260515.json),
  and interpretation notes
  [gsm8k_retry_budget_sweep_20260515_interpretation.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/gsm8k_retry_budget_sweep_20260515_interpretation.md).
  Offline budget probe on the existing T=0 `v6` rerun artifact — for `k ∈ {5,10,15,20,25,30,40,50,60}`, take the bottom-`k` flagged samples by `logistic_broad_score`, apply the archived rerun answer only on those, keep `exp_only` elsewhere.
- Readout (single rerun artifact, T=0):
  - k=5:  `+0.38pt` (1 fix, 0 hurts)
  - k=10: `+0.38pt` (2 fixes, 1 hurt)
  - k=20: `+0.76pt` (3 fixes, 1 hurt)
  - k=30: `+1.14pt` (4 fixes, 1 hurt)
  - k=40: `+1.52pt` (5 fixes, 1 hurt)
  - k=50: `+1.89pt` (6 fixes, 1 hurt)
  - **k=60 (current tau=0.5845): `+2.65pt` (8 fixes, 1 hurt)**
- Archived read:
  - fixes grow monotonically with `k`; hurts stay flat at `1` for all `k ≥ 10` (a single sample contributes the only hurt)
  - fix/hurt ratio improves with `k` (2.0 at k=10 → 8.0 at k=60)
  - **no diminishing-returns signal up to k=60** within this archived rerun artifact
- Cross-task comparison with SVAMP-tuned q25 vs q40:
  - SVAMP q25 → q40 (`n=14→23`): fixes `1→2`, hurts `0→1`, net stays `+1` — wider budget loses efficiency
  - GSM8K k=50 → k=60: fixes `6→8`, hurts `1→1`, net `+5 → +7` — wider budget keeps adding value
  - Same score, opposite archived rerun-scaling behavior — task structure (base-accuracy distribution) governs whether the score's flagged set keeps yielding flips (GSM8K) or quickly saturates (SVAMP)
- Interpretation boundary:
  - this section should not be read as a deployable budget curve for a real deterministic `T=0` retry mechanism
  - it is only a record of how one non-identical rerun artifact behaved under different offline subset sizes
- Caveats:
  - single retry artifact (v6, single seed); the `hurts=1` flat pattern is single-seed
  - cannot probe `k > 60` without new GPU runs on score-borderline samples
  - subset-base-acc is non-monotonic at very small `k` (small-N artifact)

## Phase E-Retry-Pool Status — separate self-consistency probe, gain is weak

> Section order note: this is the diversity-side question, not the targeting-side question. It is best read as a self-consistency-style probe layered on top of the main T=0 selective-retry result, not as core evidence that the reliability score itself is stronger.

- `2026-05-14`: vote-method pool analysis completed from
  [retry_vote_pool_20260514.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/retry_vote_pool_20260514.md),
  [retry_vote_pool_20260514.json](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/retry_vote_pool_20260514.json).
  K-aggregation evaluator and full GPU handoff at
  [retry_pool_handoff_20260514.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/retry_pool_handoff_20260514.md).
- Why the earlier artifacts were not a real K-pool:
  - the existing retry artifacts (`v6` exp, `v1` prob, random_60_seed123 exp) share identical raw generations on the flagged 60 (`60/60` text-equal across pairs) because the retry pipeline ran at `temperature=0.0` (deterministic generation)
  - they are therefore a single retry trajectory read off multiple ways (different voting, different cross-run aggregation), not multiple independent retry trajectories
- Cheap analog (vote-method pool over the single deterministic trajectory):
  - any single read (`v6_exp_only`, `v6_vote`, `v1_exp_only`, `v1_vote`) gives `25/60` correct on the flagged 60 (`41.67%`)
  - union-any across the four reads: `26/60` (`43.33%`) — `+1` sample rescued by vote-method choice on the same trajectory (sample `841`)
  - intersect-all across the four reads: `24/60`
  - full-test ceiling if an oracle picked the right read on each flagged sample: `70.83%` (`+0.75pt` over P_selective `70.08%`)
- Secondary finding (cross-run vote stochasticity):
  - the random retry run and the v6 retry run share `12` overlap samples and produce identical raw generations on all `12/12`
  - but their `vote_answer` differs on `8/12` of those samples — numerical non-determinism in the voting machinery across separate GPU runs at T=0
  - this is interesting structural noise, not a real trajectory diversity source for retry rescue
- Real T>0 pilot (`2026-05-15`):
  - seed plumbing added to the retry path so temperature-based reruns can actually differ
  - `T=0.5` collapsed immediately at K=1 (`0%`), so the usable pilot was rerun at `T=0.2`
  - setup: same GSM8K flagged 60, `K=3`, seeds `42/43/44`, answer_kind `exp_only`
  - artifacts:
    - [retry_pool_eval_gsm8k_t02_k3_20260515.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/retry_pool_eval_gsm8k_t02_k3_20260515.md)
    - [retry_pool_revote_eval_gsm8k_t02_k3_20260515.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/retry_pool_revote_eval_gsm8k_t02_k3_20260515.md)
- Answer-diversity sanity:
  - seed43 vs ref(seed42): `49/60` different answers
  - seed44 vs ref(seed42): `45/60` different answers
  - so this is not a hidden deterministic repeat; answer-level regeneration diversity is present
- Simple majority across runs:
  - flagged 60 majority acc: `40.00% → 40.00% → 43.33%` for `K=1/2/3`
  - union-any ceiling: `40.00% → 46.67% → 58.33%`
  - full-test deploy (majority on flagged, base elsewhere): `70.08% / 69.70% / 70.45%`
  - read: K=3 is only `+0.37pt` over the deterministic single-retry policy (`70.08%`)
- Step-level pooled re-vote:
  - merge valid events across the first K runs and re-run the original `exp_only` vote over the pooled events
  - full-test deploy: `70.08% / 68.56% / 68.56%`
  - read: pooled re-vote is worse than simple majority here
- Alternative aggregators on the same K=3 artifacts:
  - [retry_pool_aggregators_K3_20260515.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/retry_pool_aggregators_K3_20260515.md),
    [retry_pool_aggregators_K3_20260515.json](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/retry_pool_aggregators_K3_20260515.json)
  - full-test deploy:
    - `best_by_margin`: `68.56%`
    - `best_by_top1`: `68.56%`
    - `confidence_weighted`: `68.94%`
    - `two_of_K_else_base`: `70.08%`
    - `majority`: `70.45%`
  - `9` flagged samples are oracle-hit / majority-miss. The confidence-based selectors recover only a small part of this gap.
  - main negative mechanism finding: **within-retry vote confidence is not a reliable selector of which retry is correct**
- Current conclusion:
  - regeneration diversity exists
  - but current pool aggregators capture little of it
  - this branch is therefore **weakly positive at best**, and its main value is a negative self-consistency mechanism result rather than stronger support for the main reliability story
  - the line is lower priority than the main `T=0` reliability / fallback story

## Phase E-Diff2 Status (multi-seed robustness) — partially reverses E-Diff

- `2026-05-14`: multi-seed pass over 10 outer seeds completed from
  [reliability_multiseed_20260514.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/reliability_multiseed_20260514.md),
  [reliability_multiseed_20260514.json](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/reliability_multiseed_20260514.json),
  and interpretation notes
  [reliability_multiseed_20260514_interpretation.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/reliability_multiseed_20260514_interpretation.md).
- Setup:
  - same logistic_broad fit pipeline as E2/E4/E6, refit per seed
  - threshold tau picked at val 25%-quantile of the reliability score per seed
  - test reports `exp_only`, global `prob_vote`, and gated acc
- Readout (mean ± std over 10 seeds, gsm8k test):
  - `exp_only`: `69.05% ± 1.96pt`
  - `prob_vote`: `68.94% ± 1.93pt`
  - `gated`: `69.66% ± 1.82pt`
  - `gated − exp_only`: `+0.61pt ± 0.78pt`; gated wins on 70% of seeds
  - `prob_vote − exp_only`: `−0.11pt ± 1.55pt`; prob_vote globally is neutral on average
  - `gated − prob_vote`: `+0.72pt ± 1.11pt`; gated beats global swap on 70% of seeds
- Current read:
  - the E4/E6 single-seed `+1.52pt` was inflated; the true mean lift is closer to `+0.61pt`, but still positive and consistent in sign
  - the E-Diff single-seed claim that "global `prob_vote` dominates the gated rule" was also driven by seed=42's `exp_better=0`; it does not generalize
  - reconciliation with `Pearson(score, delta) ≈ 0`: gating helps not because targeting is strong, but because budget-bounded swap caps the downside; on splits where `exp_better > prob_better` (most seeds), a global swap pays for those mismatches while the gated rule's bounded budget does not
  - the threshold rule is reinstated as a legitimate (small) operational improvement on the current artifact; it is not the load-bearing finding (Phase E abstention AURC remains that)

## Phase E-Diff Status (differential targetability followup) — reframes E4/E6

- `2026-05-14`: differential targetability analysis completed from
  [reliability_differential_20260514.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/reliability_differential_20260514.md),
  [reliability_differential_20260514.json](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/reliability_differential_20260514.json),
  and interpretation notes
  [reliability_differential_20260514_interpretation.md](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/reliability_differential_20260514_interpretation.md).
- Setup:
  - for each sample compute `delta = is_correct(prob_vote) − is_correct(exp_only) ∈ {−1, 0, +1}`
  - compare reliability-score-gated swap against random-gated and oracle-gated swap at the same budget
  - check `Pearson(score, delta)` separately from `Pearson(score, y_exp)`
- Readout:
  - `Pearson(score=logistic_broad, delta)` is `−0.003` on val, `−0.129` on test, `−0.029` on SVAMP — essentially zero
  - `Pearson(score, y_exp) ≈ +0.45` and `Pearson(score, y_prob) ≈ +0.45` — score predicts both methods' correctness similarly, i.e. it captures generic difficulty
  - on gsm8k_test at budget 25%, score-gated catches 4 of 7 fixes (random expectation ~1.75), so there is a small targeting effect; but global `prob_vote` captures all 7 fixes at zero cost, dominating the gated policy
  - on gsm8k_val (the fit split), score-gated swap is at most equal to random-gated at budget 25%, and is strictly worse than random at budgets 30–50% — the targeting effect is not robust on the fit data
  - sample-level breakdown shows gsm8k_test has `exp_better=0`, `prob_better=7`, while gsm8k_val has `exp_better=23`, `prob_better=12` — the val/test mismatch is the primary driver of the E6 rule's apparent success
- Current read:
  - the reliability score is a difficulty score (abstention-grade) but not a method-switch score
  - the E4/E6 rule's `+1.52pt` lift is largely a distributional artifact of the single seed=42 test split plus a global `prob_vote` advantage on that split, with only a small targeting contribution
  - the load-bearing positive finding in the reliability track remains Phase E's abstention AURC, which does not depend on which voting method is being compared
  - operational claims involving `prob_vote` swap should be re-evaluated under a multi-seed outer split before being promoted further

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
