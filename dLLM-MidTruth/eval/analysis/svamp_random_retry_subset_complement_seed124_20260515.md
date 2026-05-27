# SVAMP Random Retry Control Subset — 20260515

- subset type: **complement-only** (overlap with flagged = `0/49`)
- k: `49` (matches the SVAMP selective retry budget at `tau=0.5845`)
- random seed: `124`
- pool size: `251` (unflagged only)
- random subset base acc: `93.88%` (SVAMP overall is `86.33%`, flagged is `55.10%`)
- json: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/svamp_random_retry_subset_complement_seed124_20260515.json`
- txt:  `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/svamp_random_retry_subset_complement_seed124_20260515.txt`

## Use — GPU random retry on this subset

```
cd /home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval

PYTHON_BIN=/home/work/GFlowPO/anaconda3/envs/prophet/bin/python \
SUBSET_INDICES_FILE=/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/svamp_random_retry_subset_complement_seed124_20260515.txt \
RUN_NAME=20260515_svamp_retry_exp_complement_seed124 \
TASK=svamp \
VOTE_METHOD=exp \
bash scripts/run_retry_policy_experiment.sh 1
```

## First 10 selected SVAMP samples

- idx `1` | score `0.9072` (flagged `N`) | base `4.0` | correct `1`
- idx `4` | score `0.8300` (flagged `N`) | base `31.0` | correct `1`
- idx `6` | score `0.8473` (flagged `N`) | base `21.0` | correct `1`
- idx `10` | score `0.9064` (flagged `N`) | base `3.0` | correct `1`
- idx `21` | score `0.6395` (flagged `N`) | base `33.0` | correct `1`
- idx `33` | score `0.9222` (flagged `N`) | base `16.0` | correct `1`
- idx `36` | score `0.6467` (flagged `N`) | base `3.0` | correct `1`
- idx `37` | score `0.8425` (flagged `N`) | base `125.0` | correct `1`
- idx `46` | score `0.6978` (flagged `N`) | base `17.0` | correct `1`
- idx `52` | score `0.8259` (flagged `N`) | base `17.0` | correct `1`
