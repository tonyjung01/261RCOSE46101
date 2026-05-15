# SVAMP-tuned Random Retry Control Subset — 20260515

- subset type: **complement random** (overlap with SVAMP-tuned flagged = `0/23`)
- score fit source: SVAMP val (seed=42 outer, 80/20)
- tau (from selective manifest): `0.9192` (val 40%-quantile)
- k: `23`
- random seed: `124`
- SVAMP test n: `60`  pool size: `37`
- random subset base acc: `100.00%`

## Use — GPU random retry on this subset

```
cd /home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval

PYTHON_BIN=/home/work/GFlowPO/anaconda3/envs/prophet/bin/python \
SUBSET_INDICES_FILE=/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/svamp_tuned_random_retry_subset_complement_seed124_q40_20260515.txt \
RUN_NAME=20260515_svamp_retry_exp_svamptuned_complement_seed124 \
TASK=svamp \
VOTE_METHOD=exp \
bash scripts/run_retry_policy_experiment.sh 1
```
