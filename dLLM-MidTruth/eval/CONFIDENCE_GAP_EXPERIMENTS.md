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
   - Verify whether confidence gap carries usable within-sample ranking signal before prioritizing hybrid voting
3. **[Phase 4]** Char-offset localization patch for Math500 AWNF
   - `_locate_answer_window_v2()` using tokenizer `offset_mapping` instead of BPE subsequence search
   - Goal: check whether Math500 AWNF can be reduced meaningfully, especially on Bucket B cases
4. **[Phase 5/5.5]** Patched debug reruns — Math500 + GSM8K with `use_char_offset=True`
5. **[Phase 6]** Post-patch reanalysis
   - Re-check AWNF reduction and compare post-patch signal quality, especially on Math500
6. **[Phase 7]** Hybrid one-factor sweep on GSM8K patched data
   - Proceed only if Phase 3 suggests the gap signal is worth exploiting
   - Factor 1: quality shape (binary / clipped / tanh)
   - Factor 2: λ ∈ [0.25, 0.5, 1.0, 2.0, 5.0]
   - Factor 3: additive vs. multiplicative formula
7. **[Phase 8]** Commit-time gap exploratory
   - Compare commit-time signal against current step-level gap as a longer-horizon follow-up

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
