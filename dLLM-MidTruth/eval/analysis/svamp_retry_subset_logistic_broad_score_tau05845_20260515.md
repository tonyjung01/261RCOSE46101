# SVAMP Selective Retry Subset — 20260515

Cross-task transfer of the GSM8K-val-fitted reliability score: same `tau`, same score definition, applied to SVAMP samples. No SVAMP-specific tuning.

- base artifact: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260429_everpass_debug_bs4/svamp_gen128_steps64_vote_confidence_gap_answer_window5_mean_rawsum_bs4_debug/rank_0_generations.json`
- score fit source: GSM8K val (seed=42, frac=0.8)
- score key: `logistic_broad_score`
- tau: `0.5845` (GSM8K val 25%-quantile of logistic_broad_score)
- SVAMP overall base acc: `86.33%`
- selected (flagged): `49/300` (`16.33%`)
  - flagged base acc: `55.10%`
  - unflagged base acc: `92.43%`
- json: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/svamp_retry_subset_logistic_broad_score_tau05845_20260515.json`
- txt:  `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/svamp_retry_subset_logistic_broad_score_tau05845_20260515.txt`

## Use — GPU retry on flagged subset

```
cd /home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval

PYTHON_BIN=/home/work/GFlowPO/anaconda3/envs/prophet/bin/python \
SUBSET_INDICES_FILE=/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/svamp_retry_subset_logistic_broad_score_tau05845_20260515.txt \
RUN_NAME=20260515_svamp_retry_exp_selective_tau05845 \
TASK=svamp \
VOTE_METHOD=exp \
bash scripts/run_retry_policy_experiment.sh 1
```

Trailing `1` is the GPU index. Expected wall-time `~25s` based on the 60-sample GSM8K retry baseline (SVAMP is shorter prompts).

## Next: complement-only random control

Generate a budget-matched random control from the unflagged SVAMP slice:

```
/home/work/GFlowPO/anaconda3/envs/prophet/bin/python scripts/select_svamp_random_retry_subset.py --random-seed 124 --exclude-flagged --tau 0.5845 --k 49
```

Then run the random GPU retry on that manifest, evaluate both with `evaluate_retry_control.py` adapted for SVAMP, or use a SVAMP-aware evaluator (see follow-up script).

## First 10 selected SVAMP samples

- idx `5` | score `0.5790` | base `720.0` | prob `720.0` | correct `1`
- idx `9` | score `0.3636` | base `14150.0` | prob `14150.0` | correct `0`
- idx `15` | score `0.5219` | base `21.0` | prob `21.0` | correct `0`
- idx `16` | score `0.5809` | base `5.0` | prob `5.0` | correct `1`
- idx `17` | score `0.5257` | base `177.0` | prob `177.0` | correct `0`
- idx `19` | score `0.5548` | base `13.0` | prob `13.0` | correct `1`
- idx `30` | score `0.4291` | base `3.0` | prob `3.0` | correct `1`
- idx `32` | score `0.3863` | base `113.0` | prob `113.0` | correct `0`
- idx `34` | score `0.3971` | base `9.0` | prob `9.0` | correct `1`
- idx `43` | score `0.4537` | base `90.0` | prob `90.0` | correct `0`
