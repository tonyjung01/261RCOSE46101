# GSM8K Gap-Correctness Analysis

- Date: 2026-05-12
- Run: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260429_everpass_debug_bs4/gsm8k_gen128_steps64_vote_confidence_gap_answer_window5_mean_rawsum_bs4_debug/rank_0_generations.json`
- Stored vote method: `confidence_gap_answer_window5_mean_rawsum`
- Samples: `1319`
- Valid events: `63687`
- Gap field coverage on valid events: `63687/63687` (`all_valid_events_have_gap_values=True`)

## Unconditional

- Pearson: `0.2201` over `63687` events
- Spearman: `0.2338` over `63687` events

## Step-Conditional

- Q1 (0-25%): Pearson `0.1109`, Spearman `0.0948`, n=`15080`
- Q2 (25-50%): Pearson `0.2006`, Spearman `0.1893`, n=`14933`
- Q3 (50-75%): Pearson `0.1552`, Spearman `0.1755`, n=`15257`
- Q4 (75-100%): Pearson `0.2185`, Spearman `0.2257`, n=`18417`

## Within-Sample Pairwise Win-Rate

- Raw: `0.7625` (`277518/363952` correct>wrong pairs)
- Contiguous dedup: `0.7034` (`14883/21160` correct>wrong pairs)
- Answer-level unique: `0.8632` (`3476/4027` correct>wrong pairs)

## Notes

- This analysis treats the stored per-step `raw_weight` as the gap-derived signal and uses the saved GSM8K ground truth for correctness.
- If the run directory name suggests `everpass` but the stored `vote_method` is a confidence-gap variant, interpret the result based on the stored method metadata rather than the directory name.
