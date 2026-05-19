# Transfer Score Ablation (`top1_prob` vs `prob_margin`)

**Date**: 2026-05-16  
**Model**: `LLaDA-8B-Instruct`  
**Primary metric**: `vote_answer` accuracy under `exp` TSCV  
**Companion artifacts**:
- `outputs/LLaDA-8B-Instruct/20260515_gsm8k_full_exp_bs4_top1prob`
- `outputs/LLaDA-8B-Instruct/20260515_gsm8k_full_exp_bs4_probmargin`
- `outputs/LLaDA-8B-Instruct/20260515_svamp_full_exp_bs4_top1prob`
- `outputs/LLaDA-8B-Instruct/20260515_svamp_full_exp_bs4_probmargin`
- `outputs/LLaDA-8B-Instruct/20260515_math500_full_exp_bs4_top1prob`
- `outputs/LLaDA-8B-Instruct/20260515_math500_full_exp_bs4_probmargin`
- `outputs/LLaDA-8B-Instruct/20260515_countdown_full_exp_bs4_top1prob`
- `outputs/LLaDA-8B-Instruct/20260515_countdown_full_exp_bs4_probmargin`

## 1. Hypothesis

The current deterministic masked-diffusion decoding policy ranks active-block masked positions by:

`score_i = p_top1(i)`

where `p_top1(i)` is the largest softmax probability at masked position `i`.

The ablation hypothesis is that a **margin-based confidence signal**

`score_i = p_top1(i) - p_top2(i)`

may be a better ranking score for deciding which masked positions to transfer at each step.

Intuition:
- `p_top1` captures absolute confidence.
- `p_top1 - p_top2` captures **local decisiveness** between the best and second-best token.
- If the transfer schedule should favor positions with less local ambiguity, margin ranking may produce a better decoding trajectory even under `T=0`.

## 2. Methodology / Controls

This experiment was intentionally kept minimal and methodology-clean.

### Fixed across both runs

- `temperature = 0.0`
- `gen_length = 128`
- `diffusion_steps = 64`
- `block_length = 32`
- semi-AR block sampling
- same model checkpoint
- same prompt / same parser
- same final aggregation: `exp`-weighted TSCV
- same seed: `42`
- same batch size: `4`
- same datasets / full evaluation split

### Changed

Only the **ranking score used to select positions to transfer** inside the active block:

- **Baseline**: `transfer_score = top1_prob`
- **Ablation**: `transfer_score = prob_margin`

### Not changed

- The number of transferred tokens per step
- The set of eligible positions (active-block masked positions only)
- The filled token at each position (`argmax(logits)`)
- The parser
- The vote logic

So this is a pure decoding-policy ablation: **same deterministic framework, same compute, different transfer ranking rule**.

## 3. Smoke Check

Before the full runs, small 32-example smoke runs were used to verify that the new flag was not a no-op.

In the `bs=4`, `VOTE_METHOD=exp` smoke:

- `top1_prob`: Final `59.38%`, Vote `65.62%`
- `prob_margin`: Final `71.88%`, Vote `71.88%`

The run pair changed:

- `vote_answer`: `9 / 32`
- `final_answer`: `9 / 32`
- generated text: `30 / 32`

So the new transfer score materially changed the trajectory under `T=0`, rather than acting as a dead flag.

## 4. Full Results

### Main comparison table

| Task | n | Baseline final | Baseline vote | `prob_margin` final | `prob_margin` vote | Final delta | Vote delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| GSM8K | 1319 | 68.39% | 69.67% | 69.37% | 70.81% | +0.98pt | +1.14pt |
| SVAMP | 300 | 84.33% | 86.00% | 86.67% | 87.33% | +2.34pt | +1.33pt |
| MATH500 | 500 | 27.00% | 27.60% | 27.20% | 27.60% | +0.20pt | +0.00pt |
| Countdown | 256 | 19.53% | 23.05% | 18.36% | 23.05% | -1.17pt | +0.00pt |

### Primary read (`vote_answer`)

If `exp` TSCV is treated as the main metric, then:

- **GSM8K**: positive (`+1.14pt`)
- **SVAMP**: positive (`+1.33pt`)
- **MATH500**: neutral (`+0.00pt`)
- **Countdown**: neutral (`+0.00pt`)

### Secondary read (`final_answer`)

The final parsed string from the last generation state also improved on GSM8K and SVAMP, was nearly unchanged on MATH500, and worsened on Countdown:

- **GSM8K**: `+0.98pt`
- **SVAMP**: `+2.34pt`
- **MATH500**: `+0.20pt`
- **Countdown**: `-1.17pt`

## 5. How Much the Trajectory Changed

The ablation was not a trivial tie-breaker. It substantially changed the decoded outputs.

| Task | `vote_answer` changed | `final_answer` changed | generated text changed |
|---|---:|---:|---:|
| GSM8K | 372 / 1319 | 398 / 1319 | 1186 / 1319 |
| SVAMP | 35 / 300 | 41 / 300 | 235 / 300 |
| MATH500 | 279 / 500 | 284 / 500 | 478 / 500 |
| Countdown | 232 / 256 | 187 / 256 | 256 / 256 |

Interpretation:
- The ranking rule affects the full trajectory, not just a tiny subset of examples.
- The change is especially strong on MATH500 and Countdown, but those tasks do not convert the extra trajectory diversity into better `vote_answer` accuracy.
- GSM8K and SVAMP are the two tasks where the changed trajectory also turns into better final performance.

## 6. Interpretation

### Main positive finding

Replacing the transfer ranking score with:

`p_top1 - p_top2`

is a **real, methodology-clean improvement** over the current `p_top1` baseline on:

- **GSM8K**
- **SVAMP**

under the exact same deterministic `T=0` decoding framework.

### What this result is *not*

This is **not**:

- a parser change
- a rerun / retry effect
- a vote-weight change
- a temperature / self-consistency effect

It is a clean within-decoder ablation.

### Task-specific behavior

- **GSM8K / SVAMP**: margin-based transfer seems to help the model decide which positions are safe to lock in earlier.
- **MATH500**: the trajectory changes a lot, but parser difficulty / task hardness likely dominates; no vote-level improvement appears.
- **Countdown**: the trajectory changes completely, but the vote metric does not improve, suggesting that the new ranking perturbs the path without helping final arithmetic validity.

## 7. Practical Conclusion

On current full runs:

- `prob_margin` should be treated as a **promising replacement** for `top1_prob` on GSM8K and SVAMP.
- It is **neutral** on MATH500 and Countdown under the current setup.
- The strongest evidence is in the primary metric:
  - GSM8K `vote_answer`: `69.67% → 70.81%`
  - SVAMP `vote_answer`: `86.00% → 87.33%`

## 8. Next Checks

Reasonable next steps if we want to harden this result:

1. Add `transfer_score` into saved artifact metadata for easier auditability.
2. Log selected transfer positions per step for direct position-diff analysis.
3. Compare against the exact paper-style reference summary in one compact table.
4. If desired, test whether a hybrid score such as
   - `margin`
   - `margin / top1`
   - or `top1 * margin`
   gives further lift without changing the rest of the decoding setup.

## 9. Bottom Line

The experiment supports the following statement:

> In deterministic masked-diffusion decoding with `exp` TSCV kept fixed, using a **probability margin** (`p_top1 - p_top2`) instead of plain `p_top1` to rank active-block masked positions yields a real decoding-policy improvement on GSM8K and SVAMP, while remaining neutral on MATH500 and Countdown.
