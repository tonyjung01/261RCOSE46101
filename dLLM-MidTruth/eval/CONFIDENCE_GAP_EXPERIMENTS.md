# Confidence Gap Experiment Tracker

This file tracks the confidence-gap voting experiments for `dLLM-MidTruth`.

## Goal

- Reproduce the original TSCV (`exp`) results.
- Evaluate whether a Prophet-style confidence-gap signal can replace time-based voting weights.
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
- `confidence_gap_answer_window5_mean_rawsum`
  - Parsed-answer locate variant using the entire `window5`.
- `confidence_gap_answer_window5_mean_rawsum_skip33pct`
  - Same as above, but skips the first 33% of steps for accumulation.
- `confidence_gap_answer_window5_blockactive_prob_mean_rawsum`
  - Current blockactive/prob-gap variant.
- `strict Prophet-compatible anchor baseline`
  - `confidence_gap_anchor_window5_logit_mean_rawsum`
  - Uses `constraints_text="96:The answer is"` and Prophet-style fixed `window5` raw-logit gap.

## Current Main Results (bs4, gen128, steps64)

| Task | Final | Exp Vote | CGap v1 Vote | CGap Skip33 Vote | CGap Blockactive Prob Vote |
|---|---:|---:|---:|---:|---:|
| Countdown | 21.48 | 25.39 | 24.22 | 24.22 | 16.80 |
| GSM8K | 68.69 | 69.98 | 69.83 | 69.83 | 69.07 |
| MATH500 | 27.00 | 27.20 | 24.60 | 24.60 | 23.80 |
| SVAMP | 84.67 | 86.33 | 86.33 | 85.67 | 84.67 |

## Current Read on the Results

- `exp` remains the strongest method across all four tasks.
- `confidence_gap_answer_window5_mean_rawsum` is close on `gsm8k` and `svamp`, but weak on `countdown` and `math500`.
- `skip33` does not help materially.
- `blockactive_prob` is the most principled variant so far, but it discards too many valid answer events:
  - many correct events are skipped as `answer_window_outside_active_block`
  - this removes the repeated-consistency signal that `exp` benefits from

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

## Planned Experiments

1. Strict Prophet-compatible anchor baseline
   - fixed answer anchor: `constraints_text="96:The answer is"`
   - `anchor_offset=2`, so the confidence window starts immediately after the Prophet-style anchor prior
   - fixed `window5`
   - raw logit gap
   - mean reduction
   - no block filtering in the baseline
2. Strict anchor baseline + `active`
3. Strict anchor baseline + `blockactive`
4. Anchor-aware hybrid locate variant
5. Revisit `math500` answer-window mismatch if AWNF remains high

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
