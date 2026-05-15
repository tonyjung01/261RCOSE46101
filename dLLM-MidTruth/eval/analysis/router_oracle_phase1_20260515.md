# Phase 1 — Within-artifact Oracle Ceiling Diagnostic — 20260515

Methodology-clean probe: single T=0 base artifact, 8 candidate answers per sample, oracle ceilings under 3 lenses. Decision criteria predefined.

- base artifact: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260429_everpass_debug_bs4/gsm8k_gen128_steps64_vote_confidence_gap_answer_window5_mean_rawsum_bs4_debug/rank_0_generations.json`
- stored vote_method: `confidence_gap_answer_window5_mean_rawsum`
- batch_size: `4`  temperature: `0.0`
- total samples: `1319`  test split (seed=42, frac=0.8): `264`

## Per-readout accuracy (val + test)

| Readout | Val Acc | Test Acc |
|---|---:|---:|
| exp_only ← baseline | 70.62% | 67.42% |
| native_vote_answer | 70.05% | 68.94% |
| final_answer | 69.19% | 66.67% |
| last_valid_answer | 69.95% | 67.05% |
| late_window_majority_q25 | 70.14% | 67.42% |
| late_window_majority_q50 | 70.05% | 69.32% |
| longest_run_answer | 67.68% | 68.56% |
| most_persistent_answer | 67.01% | 68.56% |

## Val-selected single-readout deployment (one-shot test)

- val-best readout: **`exp_only`** (val_acc `70.62%`)
- test acc under this readout: **`67.42%`**
- delta vs exp_only on test: `+0.00pt`

This is the simplest possible improvement: pick a single deterministic aggregator on val, deploy on test. No routing, no learned scorer, no rerun.

## Oracle A — full unique-answer ceiling

- 187/264 samples have at least one correct candidate
- **Oracle A acc: `70.83%`** (vs exp_only baseline `67.42%`)
- Ceiling headroom: `+3.41pt`

## Oracle B — disagreement-only ceiling

- Disagreement subset: `33/264` samples with ≥2 unique candidates
- exp_only on subset: `18.18%` (6/33)
- Oracle on subset: `45.45%` (15/33)
- Subset oracle marginal: `+27.27pt`
- Full-test gain if perfect routing on this subset: `+3.41pt`

## Oracle C — provenance-restricted

| Set | Readouts | Oracle acc |
|---|---|---:|
| compact | exp_only + native_vote_answer + final_answer | `68.94%` |
| full | compact + 5 temporal candidates | `70.83%` |

**Temporal-candidates marginal headroom: `+1.89pt`**

This tells us whether the temporal persistence decoder adds genuine new headroom beyond compact 3-readout set.

## Unique-answer count distribution (test)

| # unique candidates | # samples |
|---:|---:|
| 1 | 231 |
| 2 | 23 |
| 3 | 8 |
| 4 | 2 |

## Decision under predefined termination criteria

- Option A (terminate raw-acc line): full oracle ≤ 70% **OR** temporal marginal ≤ 1pt
- Option B (proceed to Phase 2): full vs compact gain ≥ 2pt **OR** full oracle ≥ 72%

- full oracle: `70.83%`
- temporal marginal: `+1.89pt`
- **Option A triggered: `False`**
- **Option B triggered: `False`**

**Recommendation**: ambiguous — neither criterion cleanly triggered. Inspect manually.