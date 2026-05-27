# Random Retry Control Subset — 20260514

- base artifact: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260429_everpass_debug_bs4/gsm8k_gen128_steps64_vote_confidence_gap_answer_window5_mean_rawsum_bs4_debug/rank_0_generations.json`
- subset split: `test` (n=264)
- k: `60` (random retry budget)
- random seed: `124`
- exclude flagged: `True` (score key `logistic_broad_score`, tau `0.5845`)
- random pool size: `204`
- random subset base acc: `76.67%` (`46/60`)
- overlap with score-flagged set: `0`
- json: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/random_retry_subset_test_complement_seed124_20260514.json`
- txt: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/random_retry_subset_test_complement_seed124_20260514.txt`

## Use

This is the **complement-only** random control (1순위 next step): random subset is drawn from the unflagged 204 only, so overlap with the score-flagged 60 is 0. This is the strictest single-seed control for the selective-retry targeting claim.

Run the retry on this manifest:

```
cd /home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval

PYTHON_BIN=/home/work/GFlowPO/anaconda3/envs/prophet/bin/python \
SUBSET_INDICES_FILE=/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/random_retry_subset_test_complement_seed124_20260514.txt \
RUN_NAME=20260514_gsm8k_retry_exp_complement_seed124 \
VOTE_METHOD=exp \
bash scripts/run_retry_policy_experiment.sh 1
```

Trailing `1` is the GPU index; adjust to a free GPU. Expected wall-time ~30s (matches the v6 run).

Then evaluate head-to-head against the selective retry. Note: pass this complement manifest as `--random-manifest` and the resulting complement artifact as `--random-retry-artifact`:

```
/home/work/GFlowPO/anaconda3/envs/prophet/bin/python \
  /home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/scripts/evaluate_retry_control.py \
  --selective-retry-artifact /home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260514_gsm8k_retry_exp_testsubset_tau05845_v6/rank_0_generations.json \
  --selective-manifest /home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/retry_subset_gsm8k_test_logistic_broad_tau05845_20260514.json \
  --random-retry-artifact /home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260514_gsm8k_retry_exp_complement_seed124/rank_0_generations.json \
  --random-manifest /home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/random_retry_subset_test_complement_seed124_20260514.json \
  --answer-kind exp_only \
  --output-stem retry_control_complement_eval_20260514
```

## Overlap caveat

This subset is drawn from the **complement** of the score-flagged 60 (i.e. only from the unflagged 204), so the overlap with the flagged set is `0/60`. That makes this a stricter targeting control than the pure-random seed=123 version, which had `12/60` overlap with the flagged set.

The trade-off: drawing strictly from the unflagged complement means the random subset has a base accuracy near the **unflagged** mean (~78.4%), not the overall test mean (~67.4%). The retry-on-this-subset retry rate may therefore be even lower than under pure random because the subset has less retry headroom by construction. That is the right scientific control for the targeting question: "does the score know how to spend retry budget *better than picking any other 60 samples from the population*?" If selective-retry still beats this complement control, the targeting evidence is cleaner.

## Reading the result

When the eval artifact is in, compare against the existing pure-random control:

| Comparison | What it tells us |
|---|---|
| selective vs pure random (seed=123, overlap 12/60) | already done: targeting marginal `+2.27pt` on this artifact |
| selective vs complement-only (seed=124, overlap 0/60) | this run: strict targeting marginal with zero leakage from flagged samples |
| if both controls agree directionally | targeting claim becomes more robust at single-seed |
| if complement gives a smaller marginal | the pure-random number was inflated by the 12 flagged overlap samples |

## Follow-up (2순위 — multi-seed random)

Once the complement-only result is in, run 2–3 additional random controls with different `--random-seed` values (e.g. seeds `125, 126, 127`, using `--exclude-flagged` or pure random — either is fine, pick one and stay consistent). Re-evaluate each against the same selective retry. Aggregate `(P_selective − P_random)` mean and std across the random seeds; this tells us how robust the `+2.27pt` marginal is to the choice of random draw.

## First 10 selected samples

- idx `4` | score `0.9112` (flagged `N`) | base `20.0` | prob `20.0` | correct `1`
- idx `23` | score `0.6007` (flagged `N`) | base `8.0` | prob `8.0` | correct `1`
- idx `30` | score `0.8001` (flagged `N`) | base `109.0` | prob `109.0` | correct `1`
- idx `61` | score `0.8802` (flagged `N`) | base `1430.0` | prob `1430.0` | correct `1`
- idx `65` | score `0.9183` (flagged `N`) | base `36.0` | prob `36.0` | correct `1`
- idx `118` | score `0.9378` (flagged `N`) | base `4.0` | prob `4.0` | correct `1`
- idx `145` | score `0.7975` (flagged `N`) | base `400.0` | prob `400.0` | correct `0`
- idx `154` | score `0.8063` (flagged `N`) | base `112.0` | prob `112.0` | correct `0`
- idx `163` | score `0.8260` (flagged `N`) | base `50.0` | prob `50.0` | correct `1`
- idx `175` | score `0.8751` (flagged `N`) | base `3.0` | prob `3.0` | correct `0`
