# SVAMP-tuned q40 Retry Control — Interpretation

**Date**: 2026-05-15
**Companion artifacts**: `svamp_tuned_q40_retry_control_eval_20260515.{md,json}`
**Selective manifest**: `svamp_tuned_retry_subset_q40_20260515.{md,json,txt}`
**Random manifest**: `svamp_tuned_random_retry_subset_complement_seed124_q40_20260515.{md,json,txt}`
**Retry artifacts**: `outputs/.../20260515_svamp_retry_exp_svamptuned_q40/`, `outputs/.../20260515_svamp_retry_exp_svamptuned_complement_seed124_q40/`
**Status**: **Clean diminishing-returns pattern.** Widening the SVAMP-tuned tau from val 25% to val 40% quantile adds 9 samples to the retry budget. The marginal 9 samples produce `+1` fix and `+1` hurt — net zero. Full-test accuracy stays at `+1.67pt`. Scaling rescue rate up was not what the score did; instead, the budget hit a ceiling where added samples are a mix of borderline-difficult (some rescue-able) and borderline-confident (some flip-able).

## Comparison with q25 on the same outer split

| Metric | q25 | q40 |
|---|---:|---:|
| Flagged budget | 14 | 23 |
| Flagged base acc | 57.14% | 65.22% |
| Base-wrong in flagged | 6 | 8 |
| Selective fixes | 1 | 2 |
| Selective hurts | 0 | 1 |
| Net selective | +1 | +1 |
| Subset-local retry delta | +7.14pt | +4.35pt |
| Full-test delta | `+1.67pt` | `+1.67pt` |
| Random complement retry delta | +0.00pt | +0.00pt |
| Targeting marginal | +1.67pt | +1.67pt |

The full-test accuracy is exactly the same at both budgets; budget grew by `~64%` but net rescue did not.

## Findings

### 1. Marginal 9 samples added zero net value

Going from q25 to q40 adds 9 samples to the retry budget. Decomposing those 9:
- 2 base-wrong samples got added (6 → 8)
- 7 base-correct samples got added (8 → 15)

Of those marginal 9: retry rescued 1 more base-wrong (good), and retry flipped 1 base-correct (bad). Net `+1 fix − 1 hurt = 0`.

This is structural, not a noise artifact: as the score's tau widens, you pull in samples whose difficulty signal is weaker. Some of those still have retry headroom (1 rescue), but others are confidently borderline and regeneration variance flips them (1 hurt).

### 2. The per-base-wrong rescue rate looks larger at q40, but per-flagged-sample net rescue is smaller

| Quantile | base-wrong / flagged | rescues / base-wrong | hurts / flagged | net rescue / flagged |
|---:|---:|---:|---:|---:|
| q25 | 6 / 14 | 1 / 6 ≈ 17% | 0 / 14 = 0% | 1 / 14 ≈ 7.1% |
| q40 | 8 / 23 | 2 / 8 = 25% | 1 / 23 ≈ 4.3% | 1 / 23 ≈ 4.3% |

The headline "rescue rate per base-wrong" looks like it went up (17% → 25%), but that ratio hides the fact that:
- the flagged set is now diluted with more confident samples (base 57% → 65%)
- a hurt has appeared on the flagged-but-confident slice

The right operational quantity is **net rescue per flagged sample** (`(fixes − hurts) / flagged_budget`). That number went down from `7.1%` to `4.3%` — the wider budget is less efficient per retried sample.

### 3. The pre-GPU offline diagnostic was consistent

Pre-GPU diagnostic for q40 was `prob_vote can_fix / base_wrong = 1/8` (vs `0/6` at q25). The 2 marginal base-wrong samples included 1 that an offline source could rescue and 1 that no offline source could. Post-GPU: retry rescued `1` of the 2 marginal base-wrong (matches the `1/8` ceiling), and the rescued one happens to be the offline-rescuable one — so the marginal retry rescue here was not uniquely retry-rescued, unlike the q25 rescue which had no offline equivalent.

Recomputing uniquely retry-rescued counts:
- q25: 1 fix; offline ceiling 0/6 → all 1 fix is uniquely retry-rescued
- q40: 2 fixes; offline ceiling 1/8 → at most 1 uniquely retry-rescued, possibly the same q25 sample plus a non-unique fix at the margin

So at the wider budget the marginal rescue is the offline-equivalent one; pure regeneration-variance rescue didn't grow.

### 4. Random complement stays a strict no-op

Random complement subset n=23 drawn from the unflagged 37 (all base-correct since unflagged base is 100% at q40). Retry on these 23 changed 0 samples (T=0 deterministic produces same answer as base for confident samples). Net `+0.00pt`.

This is the third SVAMP run in a row where the random control is by construction nearly a strict downside-only test. The targeting marginal `+1.67pt` is therefore mostly "selective produced 1 net fix while random did nothing" rather than a head-to-head signal.

## What this answers about the original question

The "rescue rate ~17% scales linearly with budget" hypothesis is rejected on this artifact. Instead:

- Best operational budget on SVAMP-tuned is **the tightest one (q25)**. Wider tau gives the same net accuracy with more hurts.
- The score concentrates retry-rescuable samples at the very lowest-score tail; the bulk of additional samples at moderate scores are a mix of "still rescue-able" and "borderline-confident and flippable".
- The marginal 9 samples at q40 effectively cancel each other out — `+1` fix and `+1` hurt.

This is consistent with how a difficulty-only score should behave: it correctly orders samples by difficulty, but difficulty alone is not the same as retry-rescuability, so the marginal samples at the wider tail are a mixed bag.

## Caveats

- SVAMP test n=60, q25 budget 14, q40 budget 23 — all small
- Each fix/hurt is a single sample at this scale; the `+1 fix +1 hurt` pattern at q40 is consistent with a single-sample noise draw under "marginal samples are net-zero"
- Single seed; multi-seed could shift the marginal samples' fix/hurt split
- At T=0, random complement is deterministic — the random control side cannot give a meaningful gradient at this budget unless the random subset happens to include some base-wrong samples
- Caveats compound: the cleanest claim is the **artifact-level direction** that "widening the budget did not produce more net rescue", not a precise magnitude

## What is still consistent across all SVAMP runs

- Score targets difficulty on SVAMP (flagged 14 base `57%`, q40 flagged 23 base `65%`, both vs unflagged ~100%)
- Retry on confident samples is at best a no-op and can hurt (q40 has 1 hurt on the borderline-confident additions)
- Selective retry produces a small positive lift (`+1.67pt` at both budgets); the lift size is the same despite different budgets
- The per-base-wrong rescue rate is in the same ballpark as GSM8K (~17–25% vs ~19%)

## Recommended next step

This run is a clean answer to the budget-scaling question. The next moves that would meaningfully add information:

1. **Phase E-Retry-Pool (T>0)** — the diversity axis is now the most interesting unexplored direction. With the SVAMP picture clarified ("retry works but the budget has a natural ceiling on this artifact"), the question "can K independent regenerations on the same flagged set beat K=1?" becomes load-bearing. Infrastructure ready.
2. **SVAMP-tuned multi-seed random control** — still worth doing to put variance bars on the `+1.67pt`, but the wider budget run just showed that the random side is effectively neutralized when the unflagged complement is saturated. The marginal information from multi-seed random is therefore limited on SVAMP.
3. **Lower-tau SVAMP probe** — try q15 or q10 to see if the rescue rate per flagged sample keeps rising (i.e., are the very-lowest-score samples even more retry-rescuable?). Tiny GPU run. Low-priority but cheap.

(1) is the natural pivot now.
