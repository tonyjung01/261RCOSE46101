# Random Retry Control Subset — 20260514

- base artifact: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260429_everpass_debug_bs4/gsm8k_gen128_steps64_vote_confidence_gap_answer_window5_mean_rawsum_bs4_debug/rank_0_generations.json`
- subset split: `test` (n=264)
- k: `60` (random retry budget)
- random seed: `123`
- exclude flagged: `False` (score key `logistic_broad_score`, tau `0.5845`)
- random pool size: `264`
- random subset base acc: `68.33%` (`41/60`)
- overlap with score-flagged set: `12`
- json: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/random_retry_subset_test_seed123_20260514.json`
- txt: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/random_retry_subset_test_seed123_20260514.txt`

## Use

Feed `txt` into the same retry pipeline used for the score-targeted retry. Recommended env-explicit form (matches the env used for the v6/v1 retry runs):

```
cd /home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval

PYTHON_BIN=/home/work/GFlowPO/anaconda3/envs/prophet/bin/python \
SUBSET_INDICES_FILE=/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/random_retry_subset_test_seed123_20260514.txt \
RUN_NAME=20260514_gsm8k_retry_exp_random60_seed123 \
VOTE_METHOD=exp \
bash scripts/run_retry_policy_experiment.sh 1
```

Note: trailing `1` is the GPU index. Adjust if a different GPU is free.

Then evaluate the resulting retry artifact head-to-head against the selective-retry artifact with `evaluate_retry_control.py`. Recommended env-explicit form:

```
/home/work/GFlowPO/anaconda3/envs/prophet/bin/python \
  /home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/scripts/evaluate_retry_control.py \
  --selective-retry-artifact /home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260514_gsm8k_retry_exp_testsubset_tau05845_v6/rank_0_generations.json \
  --selective-manifest /home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/retry_subset_gsm8k_test_logistic_broad_tau05845_20260514.json \
  --random-retry-artifact /home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260514_gsm8k_retry_exp_random60_seed123/rank_0_generations.json \
  --random-manifest /home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/random_retry_subset_test_seed123_20260514.json \
  --answer-kind exp_only
```

## Overlap caveat

This is a pure random control: the random subset can overlap with the score-flagged set (here `12` of `60` samples). Pure random is the simplest budget-matched control; because it may overlap with the flagged set, a complement-only random control remains a stricter optional follow-up (`select_random_retry_subset.py --exclude-flagged`).

## First 10 selected samples

- idx `1` | score `0.9479` (flagged `N`) | base `3.0` | prob `3.0` | correct `1`
- idx `4` | score `0.9112` (flagged `N`) | base `20.0` | prob `20.0` | correct `1`
- idx `23` | score `0.6007` (flagged `N`) | base `8.0` | prob `8.0` | correct `1`
- idx `51` | score `0.7022` (flagged `N`) | base `5.0` | prob `5.0` | correct `1`
- idx `61` | score `0.8802` (flagged `N`) | base `1430.0` | prob `1430.0` | correct `1`
- idx `65` | score `0.9183` (flagged `N`) | base `36.0` | prob `36.0` | correct `1`
- idx `103` | score `0.9472` (flagged `N`) | base `140.0` | prob `140.0` | correct `1`
- idx `114` | score `0.9505` (flagged `N`) | base `6.0` | prob `6.0` | correct `1`
- idx `119` | score `0.3153` (flagged `Y`) | base `56000.0` | prob `56000.0` | correct `0`
- idx `120` | score `0.9339` (flagged `N`) | base `240.0` | prob `240.0` | correct `1`
