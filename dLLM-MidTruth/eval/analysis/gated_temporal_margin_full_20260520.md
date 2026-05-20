# Gated Temporal Margin Full Run (`prob_margin` vs `gated_temporal_margin`)

**Date**: 2026-05-20  
**Model**: `LLaDA-8B-Instruct`  
**Primary metric**: `vote_answer` accuracy under fixed `exp` TSCV  
**Launcher**: [launch_gated_temporal_margin_all_tasks_tmux.sh](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/scripts/launch_gated_temporal_margin_all_tasks_tmux.sh)

## 1. Purpose

This run is the first full-scale follow-up after:

- `A = top1_prob`
- `B = prob_margin`
- `C1 = temporal_margin = margin + λ · run-length stability`

`C1` failed on GSM8K smoke, so the next design was a **gated temporal extension** intended to preserve the strength of `prob_margin` while using temporal stability only when the current position-level margin is ambiguous.

## 2. Compared rules

### Baseline (`B`)

`score_i = p_top1(i) - p_top2(i)`

This is the current best token-ordering rule from the earlier full A/B pass.

### Candidate (`C-next-1`)

`score_i = margin_i + λ · stability_i · 1[margin_i < τ]`

with:

- `margin_i = p_top1(i) - p_top2(i)`
- `stability_i = runlen_i / steps_so_far_in_block`
- `λ = 0.10`
- `τ = 0.15`

Interpretation:

- keep `prob_margin` as the main ranking signal
- only use temporal stability when the local margin is small enough to count as ambiguous

## 3. Fixed controls

All of the following were held fixed across the comparison:

- `temperature = 0.0`
- `gen_length = 128`
- `diffusion_steps = 64`
- `block_length = 32`
- semi-AR block decoding
- same model checkpoint
- same parser
- same prompts
- same final vote: `VOTE_METHOD=exp`
- same seed: `42`
- same batch size: `4`
- same full evaluation splits
- same transfer count schedule
- same filled-token rule: `argmax(logits)`

So this is still a **single-variable decoder-policy ablation**.

## 4. Exact runs executed

The launcher ran all tasks sequentially:

1. `gsm8k`
2. `svamp`
3. `math500`
4. `countdown`

For each task, it ran in parallel:

- baseline: `prob_margin`
- candidate: `gated_temporal_margin`

with these output directories:

- `outputs/LLaDA-8B-Instruct/20260520_gsm8k_full_exp_bs4_probmargin`
- `outputs/LLaDA-8B-Instruct/20260520_gsm8k_full_exp_bs4_gatedtm_l010_t015`
- `outputs/LLaDA-8B-Instruct/20260520_svamp_full_exp_bs4_probmargin`
- `outputs/LLaDA-8B-Instruct/20260520_svamp_full_exp_bs4_gatedtm_l010_t015`
- `outputs/LLaDA-8B-Instruct/20260520_math500_full_exp_bs4_probmargin`
- `outputs/LLaDA-8B-Instruct/20260520_math500_full_exp_bs4_gatedtm_l010_t015`
- `outputs/LLaDA-8B-Instruct/20260520_countdown_full_exp_bs4_probmargin`
- `outputs/LLaDA-8B-Instruct/20260520_countdown_full_exp_bs4_gatedtm_l010_t015`

## 5. Full results

| Task | `prob_margin` vote | `gated_temporal_margin` vote | Vote delta | `prob_margin` final | `gated_temporal_margin` final | Final delta |
|---|---:|---:|---:|---:|---:|---:|
| GSM8K | 70.81% | 70.58% | `-0.23pt` | 69.37% | 68.84% | `-0.53pt` |
| SVAMP | 87.33% | 88.67% | `+1.34pt` | 86.67% | 88.00% | `+1.33pt` |
| MATH500 | 27.60% | 28.40% | `+0.80pt` | 27.20% | 28.00% | `+0.80pt` |
| Countdown | 23.05% | 23.44% | `+0.39pt` | 18.36% | 19.92% | `+1.56pt` |

## 6. Read

The result is **not** a universal improvement.

### What improved

- `SVAMP`: clear win
- `MATH500`: modest win
- `Countdown`: small vote gain, larger final gain

### What regressed

- `GSM8K`: small but real regression on both vote and final

## 7. Main interpretation

This means the first gated temporal redesign is a **cross-task tradeoff**, not a clean replacement for `prob_margin`.

The current practical read is:

- `prob_margin` remains the best **GSM8K-first** rule
- `gated_temporal_margin` is a promising **task-dependent variant**
- the temporal signal is not useless, but it is still not aligned enough to beat `prob_margin` on the main reference task

## 8. Bottom line

The clean statement from this run is:

> Gating temporal stability by low local margin is materially better than the naive `C1` run-length extension, but it still does not dominate `prob_margin`: it improves SVAMP, MATH500, and Countdown while slightly hurting GSM8K.

So the current ranking is:

1. `prob_margin` — best current default for the original Layer-4 claim
2. `gated_temporal_margin` — useful follow-up with broader but non-uniform gains
3. `temporal_margin (C1)` — negative first temporal extension
