# Token Ordering Experiment

**Date**: 2026-05-20  
**Scope**: adaptive token ordering for masked-diffusion decoding in `LLaDA-8B-Instruct` on math-style benchmarks  
**Status**: `A/B` complete, `C1` smoke complete, redesign in progress

---

## 1. Goal

This experiment line asks a different question from the earlier cgap voting work.

Earlier cgap experiments mostly asked:

> can confidence-gap signals improve **answer selection / vote aggregation**?

This new line asks:

> can confidence or margin signals improve **which masked token positions are opened first** during decoding?

So this is **not** another voting-method experiment.
It is a **decoder policy** experiment.

---

## 2. Literature positioning

The three relevant ideas live at different levels of the generation pipeline.

| Work | Unit of analysis | Main idea |
|---|---|---|
| `Time Is a Feature` | answer / trajectory | correct answers may appear in intermediate steps and later get overwritten, so stepwise parsed answers should be aggregated |
| `Prophet` | answer region / commit timing | a strong confidence-gap can indicate that the answer is ready to commit |
| `Kim et al.` | token position | masked-diffusion performance depends strongly on **which positions are unmasked first** |

This project's new direction is closest to **Kim et al.**

The mapping is:

- `Time Is a Feature`: how should stepwise answers be aggregated?
- `Prophet`: when should an answer or answer region be treated as stable enough to commit?
- `Kim et al.`: which masked token positions should be opened first?

So the idea we borrow here is:

> instead of changing answer voting again, change the **token unmask order** that creates the answer trajectory in the first place.

---

## 3. Decoder view of the current baseline

Current fixed setup:

- `T = 0`
- `gen_length = 128`
- `diffusion_steps = 64`
- `block_length = 32`
- semi-AR block sampling
- top-k confidence mask transfer
- final aggregation: `exp`-weighted TSCV
- parser unchanged
- prompt unchanged
- no rerun / no setup mismatch

At a high level, the current baseline does:

1. compute logits at each step
2. choose a subset of active-block masked positions to fill
3. fill those positions with `argmax(logits)`
4. repeat
5. collect stepwise parsed answers
6. apply `exp`-weighted TSCV

The current transfer score is:

`score_i = p_top1(i)`

where `p_top1(i)` is the largest softmax probability at masked position `i`.

---

## 4. New experiment ladder

We want a minimal progression:

- `A`: `top1_prob` transfer + `exp` TSCV
- `B`: `prob_margin` transfer + `exp` TSCV
- `C`: `temporal_margin` transfer + `exp` TSCV

### A. `top1_prob`

Baseline transfer policy:

`score_i = p_top1(i)`

### B. `prob_margin`

Kim-style probability margin transfer:

`score_i = p_top1(i) - p_top2(i)`

This changes only the **ranking of masked positions**.

It does **not** change:

- transfer token count per step
- eligible positions
- filled token value (`argmax`)
- parser
- final `exp` TSCV vote

### C. `temporal_margin` (planned)

Planned extension:

`score_i = margin_i + λ · stability_i`

where:

- `margin_i = p_top1(i) - p_top2(i)`
- `stability_i` = how stable the top-1 token at position `i` has been across recent steps

This is the point where the line becomes more than a straight Kim replication.

Interpretation:

- `Kim et al.`: open less ambiguous positions first
- `Prophet`: confidence-gap may indicate commit readiness
- `Time Is a Feature`: temporally stable intermediate signals are useful

So the intended C-stage idea is:

> prefer positions that are both locally decisive **and** temporally stable.

---

## 5. Hypotheses

### H1. Kim-style margin helps

If `B > A`, then simple margin-based token ordering is useful in this math decoding setting.

### H2. Temporal stability adds signal

If `C > B`, then the temporal viewpoint from `Time Is a Feature` / `Prophet` adds useful information beyond current-step margin alone.

### H3. Negative case

If `B ≈ A` and `C ≈ B`, then confidence/margin-based token ordering is weak in the current `LLaDA + exp-TSCV` setup.

---

## 6. Methodology guarantees

This line is meant to stay clean.

- same deterministic `T=0` setup
- same parser
- same prompts
- same generation length / diffusion steps / block length
- same model checkpoint
- same final vote rule
- no rerun-based comparison
- no setup mismatch

So if a gain appears, it should be interpretable as a **within-decoder policy gain**.

---

## 7. Results so far: A vs B

`A` and `B` have already been run with the same `exp` voting framework.

Primary metric is `vote_answer` accuracy, since the target method still ends in `exp`-weighted TSCV.

### Full-run comparison

| Task | n | A: `top1_prob` vote acc | B: `prob_margin` vote acc | delta |
|---|---:|---:|---:|---:|
| GSM8K | 1319 | 69.67% | 70.81% | `+1.14pt` |
| SVAMP | 300 | 86.00% | 87.33% | `+1.33pt` |
| MATH500 | 500 | 27.60% | 27.60% | `+0.00pt` |
| Countdown | 256 | 23.05% | 23.05% | `+0.00pt` |

### Secondary metric (`final_answer`)

| Task | A: `top1_prob` final acc | B: `prob_margin` final acc | delta |
|---|---:|---:|---:|
| GSM8K | 68.39% | 69.37% | `+0.98pt` |
| SVAMP | 84.33% | 86.67% | `+2.34pt` |
| MATH500 | 27.00% | 27.20% | `+0.20pt` |
| Countdown | 19.53% | 18.36% | `-1.17pt` |

### Read so far

- `B > A` on **GSM8K** and **SVAMP**
- `B ≈ A` on **MATH500**
- `B` changes the trajectory a lot on **Countdown**, but does not improve the main vote metric

So the current evidence supports:

> Kim-style margin ordering is a promising positive signal for GSM8K and SVAMP under the current deterministic decoding framework.

---

## 8. Why this is different from earlier cgap failures

Earlier cgap experiments mostly tried:

`gap -> answer vote weight`

This line instead uses:

`gap / margin -> masked token position ordering`

That distinction matters.

The earlier null result does **not** automatically imply this line should fail, because the signal is now used at a lower level of the pipeline: **trajectory construction**, not **answer aggregation**.

---

## 9. First temporal extension: `C1` smoke result

The first temporal extension has now been implemented and smoke-tested on GSM8K.

### `C1` definition

`score_i(t) = margin_i(t) + λ · stability_i(t)`

with:

- `margin_i(t) = p_top1(i,t) - p_top2(i,t)`
- `stability_i(t) = runlen_i(t) / (t + 1)`
- `runlen_i(t)` = consecutive recent steps for which the position-level top-1 token has stayed unchanged

This is the simplest temporal version of the score:

> prefer positions that are both locally decisive and recently stable.

### GSM8K smoke result (`n = 64`, vote metric primary)

| condition | vote acc | final acc |
|---|---:|---:|
| `A = top1_prob` | 76.56% | 76.56% |
| `B = prob_margin` | **79.69%** | **78.12%** |
| `C1a = temporal_margin, λ=0.05` | 75.00% | 73.44% |
| `C1b = temporal_margin, λ=0.10` | 76.56% | 73.44% |
| `C1c = temporal_margin, λ=0.20` | 75.00% | 71.88% |

### Read

- `B` clearly remains best on the primary metric.
- The first temporal extension does **not** improve over `B`.
- On this smoke subset, `C1` is either worse than `A`/`B`, or at best ties `A` while still trailing `B`.

So the current honest read is:

> plain Kim-style probability margin is the strongest rule so far, and the naive temporal stability term used in `C1` is too blunt to provide extra lift.

This does **not** invalidate the token-ordering line.
It narrows the current conclusion:

- `B` is a real positive
- `C1` is a negative first temporal extension

---

## 10. Next step

The next natural step is no longer "run `C` directly at scale".

It is:

1. keep `B` as the current best token-ordering rule
2. treat `C1` as a negative first extension
3. redesign the temporal term using mechanism analysis before any full run

### Planned redesign direction

Keep everything fixed except token-order ranking:

- `A`: `p_top1`
- `B`: `p_top1 - p_top2`
- `C-next`: revised temporal score after overlap / selection analysis

The main interpretive table will be:

| outcome | interpretation |
|---|---|
| `B > A` | Kim-style token ordering helps |
| `C-next > B` | a better temporal signal exists beyond plain margin |
| `C1 < B` | naive temporal persistence is not sufficient |
| `C-next ≈ B` | current-step margin already captures most of the useful signal |
| `B, C-next both fail` | token-order confidence signals are weak in this setup |

---

## 11. Current project claim for this line

At the current stage, the clean claim is:

> We implemented a methodology-clean token-ordering ablation inside deterministic masked-diffusion decoding. Replacing plain top-1 probability with probability margin improves `exp`-TSCV vote accuracy on GSM8K and SVAMP, while remaining neutral on MATH500 and Countdown.

The stronger claim we wanted to test next was:

> temporal stability can further improve token ordering beyond Kim-style probability margin alone.

The first attempt at that stronger claim (`C1`) is currently negative on GSM8K smoke, so the updated status is:

> temporal extensions remain open, but the simple run-length stability term is not yet a good replacement for plain probability margin.

---

## 11. Implementation notes

### Where the current policy lives

The current transfer-ranking logic lives in:

- [generate.py](/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/generate.py)

At each decoding step:

1. logits are computed over the current partially-filled sequence
2. the active block is identified
3. masked positions inside that active block receive a ranking score
4. top-k positions under that score are transferred
5. transferred tokens are still filled by `argmax(logits)`

This means the experiment changes:

- **position ranking**

but does not change:

- token identity at transferred positions
- transfer count per step
- parser
- final `exp` TSCV

### Current `A` / `B` / `C1` implementation status

- `A`: implemented and run
- `B`: implemented and run
- `C1`: implemented and smoke-tested
- broader `C` redesign: still open

The existing implementation already supports:

- `transfer_score = "top1_prob"`
- `transfer_score = "prob_margin"`
- `transfer_score = "temporal_margin"`

So the next real implementation task is not another reproduction run, but redesigning the temporal signal beyond the current `C1` run-length form.

---

## 12. Proposed temporal redesign directions

The main design question after the negative `C1` smoke is:

> how should we define position-level temporal information in a way that is deterministic, local, cheap, and less blunt than naive run-length persistence?

Below are three progressively stronger options.

### C1. Consecutive top-1 stability

For each position `i`, track how many recent steps the current top-1 token has remained unchanged.

Notation:

- `margin_i(t) = p_top1(i,t) - p_top2(i,t)`
- `runlen_i(t) =` consecutive recent steps where the top-1 token at position `i` stayed the same

Then:

`score_i(t) = margin_i(t) + λ · normalized_runlen_i(t)`

Pros:

- simple
- interpretable
- directly tied to temporal persistence

Cons:

- only uses the identity of the top-1 token
- ignores whether the confidence is oscillating while the token stays the same
- empirically underperformed plain `prob_margin` in the first GSM8K smoke

### C2. Margin-smoothed stability

Instead of using token identity only, combine the current margin with a short moving average of past margins:

`score_i(t) = margin_i(t) + λ · mean(margin_i(t-w+1 : t))`

Pros:

- easy to implement
- uses only confidence history
- less stateful than tracking token identity changes

Cons:

- weaker connection to actual token stability
- may reward consistently mediocre confidence

### C3. Hybrid token-and-margin stability

Combine both:

`score_i(t) = margin_i(t) + λ1 · normalized_runlen_i(t) + λ2 · mean(margin_i(t-w+1 : t))`

Pros:

- most expressive
- closest to the intended “confident and stable” interpretation

Cons:

- more hyperparameters
- easier to overfit if we tune too aggressively

### Recommended next variant

`C1` was the right first test because it was easiest to explain and cheapest to implement.

But after the negative smoke, the recommended next variant is no longer "push C1 harder".

Instead:

- keep `C1` as a documented failed first extension
- inspect `selected_transfer_indices`
- redesign the temporal term using mechanism evidence

---

## 13. Minimal next-step plan after `C1`

To keep the methodology as clean as the `A/B` pass, the next temporal experiment should still change as little as possible.

### Fixed

- same task set
- same `T=0`
- same prompt
- same parser
- same `exp` TSCV
- same transfer counts
- same `gen_length = 128`
- same `diffusion_steps = 64`
- same `block_length = 32`

### New knobs

Only:

- temporal score type
- `λ`

### Immediate sequence

1. analyze `A` vs `B` vs `C1` selected transfer overlap on a debug subset
2. identify whether `C1` is perturbing too many positions, or simply rewarding stale-but-wrong positions
3. redesign the temporal term
4. rerun GSM8K smoke only
5. only if the new temporal rule beats `B`, run GSM8K full and then SVAMP

So the current line is:

- **do not** run `C1` full
- **do** use the new logging to inspect what `C1` is doing

---

## 14. Logging / analysis needs before the redesign

The current `A/B` experiment was enough to measure end accuracy, but `C` will benefit from stronger debugging visibility.

Useful additions:

1. Save `transfer_score` explicitly into the artifact metadata
2. Optionally log selected transfer indices per step
3. Optionally log per-position stability summaries for a small debug subset

These are not conceptually required for the algorithm, but they make it much easier to explain *why* a temporal extension helps or fails.

The most important immediate use is:

> compare `selected_transfer_indices` between `A`, `B`, and `C1` to see whether the temporal term is causing broad reordering or only a few crucial divergences.

---

## 15. Decision criteria for the redesigned temporal pass

### Positive

- redesigned temporal rule `> B` on GSM8K vote accuracy
- ideally also `> B` on SVAMP

### Weak positive

- redesigned temporal rule `≈ B` on vote accuracy but with materially cleaner trajectories or stronger final-answer stability

### Negative

- redesigned temporal rule `≈ B` everywhere
- or `< B` on GSM8K again

If the redesigned temporal pass is also negative, the honest read will be:

> Kim-style current-step margin captures most of the useful token-ordering signal, and adding temporal stability does not buy additional accuracy in the current setup.

---

## 16. Working narrative

The clean narrative for this experiment line is now:

1. Earlier cgap work showed that gap-based **answer voting** does not help.
2. We then moved the signal lower in the pipeline:
   - not answer weighting
   - but token-position ordering
3. A pure Kim-style margin rule already improves GSM8K and SVAMP.
4. A first temporal extension (`C1`) did not beat plain margin.
5. The next question is whether a **better** temporal formulation can refine that margin rule further.

That gives us a precise statement of novelty:

> We are not just re-running Kim et al.; we are using Kim-style token ordering as the baseline, then testing whether temporal stability signals from `Time Is a Feature` / `Prophet` can further improve deterministic masked-diffusion decoding. The first simple temporal attempt failed, so the next contribution must come from a better temporal signal, not from claiming success too early.
