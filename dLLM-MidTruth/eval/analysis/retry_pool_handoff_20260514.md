# Phase E-Retry-Pool Handoff — 20260514

## Status

The vote-method pool over the existing single-trajectory retry is essentially exhausted: at temperature=0.0 the retry trajectory is deterministic, so `v6` (exp voting), `v1` (prob voting), and the 12-sample overlap with the random retry run all share identical raw generations on the flagged 60. The only diversity that emerges from those artifacts is vote-aggregation noise across separate GPU runs (`8/12` vote disagreements on identical trajectories), which is not what Phase E-Retry-Pool is asking about.

Real Phase E-Retry-Pool needs `K` independent retry trajectories. With temperature=0.0 those are all the same; you need either `temperature > 0` or a different sampling seed each run.

## What is already done (offline)

- `eval/analysis/retry_vote_pool_20260514.{md,json}` — vote-method pool over the single deterministic trajectory; union-any ceiling on the flagged 60 reaches `26/60` (`+1` over any single read).
- `eval/scripts/evaluate_retry_pool.py` — K-aggregation evaluator (union-any, majority, new-rescue marginal, full-test deploy). It is ready to consume `K` real retry artifacts.

## What needs to run on the GPU

`K ≥ 2` regeneration retries on the same flagged subset, at temperature > 0 (recommended: try `K=3` with `T ∈ {0.5, 0.7, 1.0}` and see if `new_rescue` keeps growing).

Manifest (already exists):

```
/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/retry_subset_gsm8k_test_logistic_broad_tau05845_20260514.txt
```

Suggested K=3 runs (`T=0.5`, three different naming suffixes to denote runs k1/k2/k3):

```
cd /home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval

# k1
PYTHON_BIN=/home/work/GFlowPO/anaconda3/envs/prophet/bin/python \
SUBSET_INDICES_FILE=/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/retry_subset_gsm8k_test_logistic_broad_tau05845_20260514.txt \
TEMPERATURE=0.5 \
RUN_NAME=20260514_gsm8k_retry_exp_pool_T05_k1 \
VOTE_METHOD=exp \
bash scripts/run_retry_policy_experiment.sh 1

# k2
PYTHON_BIN=/home/work/GFlowPO/anaconda3/envs/prophet/bin/python \
SUBSET_INDICES_FILE=/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/retry_subset_gsm8k_test_logistic_broad_tau05845_20260514.txt \
TEMPERATURE=0.5 \
RUN_NAME=20260514_gsm8k_retry_exp_pool_T05_k2 \
VOTE_METHOD=exp \
bash scripts/run_retry_policy_experiment.sh 1

# k3
PYTHON_BIN=/home/work/GFlowPO/anaconda3/envs/prophet/bin/python \
SUBSET_INDICES_FILE=/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/retry_subset_gsm8k_test_logistic_broad_tau05845_20260514.txt \
TEMPERATURE=0.5 \
RUN_NAME=20260514_gsm8k_retry_exp_pool_T05_k3 \
VOTE_METHOD=exp \
bash scripts/run_retry_policy_experiment.sh 1
```

Trailing `1` is the GPU index — adjust per availability.

Each run is `~30s` based on the v6 timing.

## Caveats / decisions to make before running

1. **Temperature changes the retry comparison baseline.** The existing selective-retry artifact (`v6`) was T=0. If you run the K-pool at T=0.5, the comparison "K=1 at T=0.5 vs T=0 retry" mixes two effects. Cleanest read:
   - Treat each new T>0 run as one of `K` independent trajectories; compare `K-pool result at T=0.5` against `K=1 at T=0.5` (one of the new runs as a baseline), not against `v6`.
   - Or: also include `v6` (T=0) as a special K=0 / control point in the same evaluator, accepting that it's a different distribution.

2. **Seed determinism check.** If two T=0.5 runs come back with identical raw generations, the pipeline is not seeding randomness per-run (rare but possible). The evaluator's `trajectory diversity sanity` row will flag this immediately (`text diff` should be > 0 across runs). If it shows 0, additional seed manipulation is needed.

3. **K choice.** Start with K=3. If new_rescue is still positive at K=3 (the third run still adds fixes the first two didn't), K=5 is worth running. If new_rescue drops to 0 quickly, diversity is saturating and bigger K is wasteful.

## Once the K artifacts exist

Run (single line, with `--retry-artifact` repeated K times):

```
/home/work/GFlowPO/anaconda3/envs/prophet/bin/python /home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/scripts/evaluate_retry_pool.py --retry-artifact /home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260514_gsm8k_retry_exp_pool_T05_k1/rank_0_generations.json --retry-artifact /home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260514_gsm8k_retry_exp_pool_T05_k2/rank_0_generations.json --retry-artifact /home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260514_gsm8k_retry_exp_pool_T05_k3/rank_0_generations.json --answer-kind exp_only --output-stem retry_pool_eval_K3_T05_20260514
```

Output:
- `eval/analysis/retry_pool_eval_K3_T05_20260514.{md,json}`
- per-K cumulative union-any / majority / new-rescue / full-test deploy table
- trajectory diversity sanity row at top (verify text-diff > 0)

## Reading the result

| Pattern | Interpretation |
|---|---|
| `new_rescue` stays positive at K=2 and K=3, plateaus by K=5 | Real regeneration diversity, exploitable up to a saturation point. K=3 is a practical operating point. |
| `new_rescue` collapses to 0 at K=2 | Trajectories are similar; T=0.5 isn't producing enough diversity. Try higher T or different sampling. |
| `majority(K) full-test acc` grows monotonically with K | Deployable diversity gain. Cost: K× retry GPU. |
| `majority(K) full-test acc` ≤ K=1 acc | Majority is harming via voting on noisy reads; stick with K=1 or union with an oracle picker. |
| `union-any` significantly > `majority(K)` | There is exploitable diversity but no good aggregator — could motivate a learned picker. |
