# Transfer Overlap Analysis (`GSM8K` smoke, `n=64`)

**Date**: 2026-05-20  
**Scope**: compare step-level `selected_transfer_indices` overlap across token-ordering policies  
**Runs**:
- `A = top1_prob`
- `B = prob_margin`
- `C1a = temporal_margin, λ=0.05`
- `C1b = temporal_margin, λ=0.10`
- `C1c = temporal_margin, λ=0.20`

## 1. Goal

The overlap analysis asks a mechanism question:

> when token-ordering performance changes, is it because the policy makes a small number of crucial selection changes, or because it broadly perturbs the transfer order?

This is especially important after the first temporal extension (`C1`) failed to beat `B = prob_margin`.

## 2. What was compared

For each sample and each decoding step, we compared:

- `selected_transfer_indices`
- step-level Jaccard overlap
- exact-match rate
- earliest divergence step
- symmetric-difference size

We also grouped samples by vote-level outcome:

- `improved`
- `hurt`
- `same_correct`
- `same_wrong`

## 3. Key headline

The main result is:

> `B = prob_margin` changes token ordering very early and very strongly relative to `A`, and that is where the positive lift comes from.
>
> `C1` changes `B` less aggressively, but in a way that mostly removes `B`'s gain rather than extending it.

So the current read is **not** that `C1` fails because it is too weak to matter.
It does matter. It just does not matter in a useful direction.

## 4. Pairwise summary

### `A` vs `B`

- changed sample fraction: `1.0000`
- changed step fraction: `0.7532`
- mean step Jaccard: `0.3433`
- exact step match rate: `0.2468`
- mean earliest divergence step: `4.75`
- vote changed count: `12`
- final changed count: `14`

Interpretation:

- `B` is **not** a minor tie-break.
- It reorders transfer selections very broadly and very early.
- That strong early reordering is consistent with `B` being a genuinely different decoding policy.

### `B` vs `C1a` (`λ=0.05`)

- changed sample fraction: `1.0000`
- changed step fraction: `0.4087`
- mean step Jaccard: `0.6571`
- exact step match rate: `0.5913`
- mean earliest divergence step: `11.52`
- vote changed count: `5`
- outcome split: `0 improved / 3 hurt`

### `B` vs `C1b` (`λ=0.10`)

- changed sample fraction: `1.0000`
- changed step fraction: `0.5010`
- mean step Jaccard: `0.5751`
- exact step match rate: `0.4990`
- mean earliest divergence step: `9.59`
- vote changed count: `5`
- outcome split: `0 improved / 2 hurt`

### `B` vs `C1c` (`λ=0.20`)

- changed sample fraction: `1.0000`
- changed step fraction: `0.6594`
- mean step Jaccard: `0.4255`
- exact step match rate: `0.3406`
- mean earliest divergence step: `6.48`
- vote changed count: `12`
- outcome split: `1 improved / 4 hurt`

Interpretation across the `C1` sweep:

- increasing `λ` makes `C1` more disruptive relative to `B`
- disruption starts earlier as `λ` increases
- harms grow faster than gains

This matches the smoke accuracy result:

- `B` remains best
- `C1a/C1b/C1c` all underperform `B`

## 5. Important structural finding: `union_jaccard = 1.0`

Across all pairwise comparisons above, sample-level union Jaccard was effectively `1.0`.

This means:

> the policies are not changing **which positions eventually get filled**.
> They are changing **when those positions get filled**.

That is exactly what we want from a token-ordering analysis:

- the mechanism difference is in **order**
- not in coverage

So the performance differences are genuinely about **transfer scheduling**, not about some positions being skipped entirely.

## 6. What the outcome groups say

### `A -> B`

For the `improved` group:

- mean step Jaccard: `0.1953`
- exact step rate: `0.1055`
- mean earliest divergence step: `1.75`

For the `hurt` group:

- mean step Jaccard: `0.2031`
- exact step rate: `0.0703`
- mean earliest divergence step: `3.00`

Interpretation:

- the samples where `B` helps are among the **most strongly reordered**
- those gains happen on samples that diverge **very early**

So `B`'s lift appears to come from decisive early scheduling changes, not from small late corrections.

### `B -> C1`

For `C1a`, `C1b`, and `C1c`, there are essentially no meaningful improvement groups.
The temporal variants mostly produce:

- `same_correct`
- `same_wrong`
- small numbers of `hurt`

Interpretation:

- `C1` is not discovering a new positive slice that `B` missed
- instead, it mostly perturbs already-good `B` trajectories or leaves them unchanged

## 7. Mechanism read

The overlap result supports the following explanation:

1. `prob_margin` is useful because it performs **strong early reordering**
2. the naive `run-length` stability term in `C1` does not add a complementary signal
3. instead, it likely rewards positions whose top-1 token is merely **persistent**, not necessarily **helpfully decisive**

This is why the current `C1` variants look bad:

- they are not just “too weak to matter”
- they modify the transfer order in a real way
- but the induced modifications are not aligned with better vote accuracy

## 8. Design implication

The next step should **not** be a full `C1` run.

The next step should be a redesigned temporal signal with a more selective role.

Examples of more plausible redesign directions:

- use temporal information only as a **tie-break** when margin is small
- gate stability by uncertainty, e.g. reward stability only for ambiguous positions
- use **margin stability** instead of raw top-1 identity persistence

The key requirement is:

> temporal information must refine margin ordering, not bluntly compete with it.

## 9. Bottom line

The overlap analysis clarifies the C1 failure.

It suggests:

- `B = prob_margin` is a strong, genuinely different policy
- `C1` is also a genuinely different policy
- but `C1` changes the schedule in ways that mostly erode `B`'s gain rather than extend it

So the right project state is:

- keep `B` as the current best token-ordering rule
- treat `C1` as a useful failed extension
- redesign temporal information before the next smoke run
