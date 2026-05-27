# C1 Temporal-Margin Smoke (`GSM8K`, `n=64`)

**Date**: 2026-05-20  
**Model**: `LLaDA-8B-Instruct`  
**Primary metric**: `vote_answer` accuracy under fixed `exp` TSCV  
**Purpose**: first test of a temporal extension on top of the positive `prob_margin` token-ordering result

## 1. Goal

After the clean `A/B` result:

- `A = top1_prob`
- `B = prob_margin`

the next question was:

> can temporal stability improve token ordering beyond plain probability margin?

The first temporal attempt (`C1`) used a simple position-level run-length signal.

## 2. Score definition

`C1` used:

`score_i(t) = margin_i(t) + λ · stability_i(t)`

where:

- `margin_i(t) = p_top1(i,t) - p_top2(i,t)`
- `stability_i(t) = runlen_i(t) / (t + 1)`
- `runlen_i(t)` is the number of consecutive recent steps for which the top-1 token at position `i` remained unchanged

This keeps the rest of decoding fixed:

- same `T=0`
- same parser
- same prompt
- same generation length / diffusion steps / block length
- same `exp` TSCV vote
- same transfer count per step
- same `argmax` fill rule

## 3. Smoke setup

- task: `GSM8K`
- subset size: `64`
- batch size: `4`
- compared conditions:
  - `A = top1_prob`
  - `B = prob_margin`
  - `C1a = temporal_margin, λ = 0.05`
  - `C1b = temporal_margin, λ = 0.10`
  - `C1c = temporal_margin, λ = 0.20`

## 4. Results

| Condition | Vote acc | Final acc |
|---|---:|---:|
| `A = top1_prob` | 76.56% | 76.56% |
| `B = prob_margin` | **79.69%** | **78.12%** |
| `C1a = temporal_margin, λ=0.05` | 75.00% | 73.44% |
| `C1b = temporal_margin, λ=0.10` | 76.56% | 73.44% |
| `C1c = temporal_margin, λ=0.20` | 75.00% | 71.88% |

## 5. Read

The result is decisively negative for `C1`.

- `B` remains the strongest rule on the primary metric.
- None of the three `C1` settings improves over `B`.
- Two of the three `C1` settings are also below `A` on the vote metric.
- `λ=0.10` ties `A` on vote, but still remains well below `B`, and its final-answer metric is worse.

So the current honest conclusion is:

> the simple temporal run-length term used in `C1` does not improve token ordering beyond plain probability margin.

## 6. Interpretation

This outcome does **not** weaken the positive `B` result.

Instead, it sharpens the project state:

- `prob_margin` is the current best clean token-ordering rule
- the first temporal extension is not good enough

Two plausible mechanism reads are:

1. **Plain margin already captures most of the useful local signal**
2. **Naive run-length stability is too blunt**
   - it may reward stale-but-wrong top-1 tokens
   - it may perturb the transfer order too broadly without improving the decisive positions

## 7. Design implication

The right next step is **not** a full `C1` run.

The right next step is:

1. inspect `selected_transfer_indices` overlap between `A`, `B`, and `C1`
2. understand whether `C1` changes too many positions, or the wrong positions
3. redesign the temporal term before another full evaluation

So the updated token-ordering ladder is:

- `A`: useful baseline
- `B`: clean positive and current best
- `C1`: negative first temporal extension
- `C-next`: redesign after mechanism analysis

## 8. Bottom line

`C1` is a useful failed experiment.

It tells us that:

> temporal information is not automatically helpful just because plain probability margin helps.

The next improvement, if it exists, will need a better temporal formulation than simple top-1 run-length persistence.
