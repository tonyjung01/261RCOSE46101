# True Retry Evaluation — 20260514

- base artifact: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260429_everpass_debug_bs4/gsm8k_gen128_steps64_vote_confidence_gap_answer_window5_mean_rawsum_bs4_debug/rank_0_generations.json`
- retry artifact: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260514_gsm8k_retry_exp_testsubset_tau05845_v6/rank_0_generations.json`
- score key: `logistic_broad_score`
- tau: `0.5845`
- eval split: `test`
- retry answer kind: `exp_only`

## Retry coverage

- retry eligible: `60/264` (`22.73%`)
- retry answers available: `60`
- retry coverage among eligible: `100.00%`

## Policies

- baseline `exp_only`: `67.42%`
- P1 offline prob fallback: `68.94%` (`+1.52%`), changed `14`, fixes `4`, hurts `0`
- P2 true retry (retry exp_only): `70.08%` (`+2.65%`), changed `28`, fixes `8`, hurts `1`
- P3 true retry (exp_only): `70.08%` (`+2.65%`), changed `28`, fixes `8`, hurts `1`
