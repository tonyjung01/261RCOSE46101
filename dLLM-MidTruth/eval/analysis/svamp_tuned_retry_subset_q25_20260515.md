# SVAMP-tuned Selective Retry Subset — 20260515

Mechanism check for the SVAMP mixed-transfer result. Score refit on SVAMP val (same 80/20 outer split, same logistic_broad pipeline); retry budget defined by val 25%-quantile of the new score; selective retry applied to held-out SVAMP test.

## Setup

- score fit source: SVAMP val (240 samples), val_base_acc `86.25%`
- best L2: `0.01`, inner-dev AURC: `0.0653`
- features kept: `['max_gap', 'mean_gap', 'last_change_gap', 'first_appear_gap', 'std_gap', 'n_valid', 'n_unique_answers', 'last_change_step_frac', 'max_gap_step_frac']`
- tau: `0.8697` (val 25%-quantile)
- SVAMP test n: `60`, test_base_acc `86.67%`
- flagged on test: `14/60` (23.33%)  base_acc `57.14%`
- unflagged on test: `46/60`  base_acc `95.65%`

## Offline diagnostics on flagged test samples

- base-wrong in flagged: `6/14`
- offline rescue ceiling (`prob_vote` correct on base-wrong): `0/6`

If the offline rescue ceiling is `0/N`, no static fallback can help and retry rescue depends entirely on generation variance. If it is `k > 0`, at least `k` of the flagged samples are recoverable by an existing offline source.

## Comparison with the GSM8K-transferred flagged set (on the same SVAMP test split)

| | n |
|---|---:|
| SVAMP-tuned flagged | 14 |
| GSM8K-transferred flagged (tau=0.5845) | 8 |
| overlap | 7 |
| SVAMP-only | 7 |
| GSM8K-only | 1 |

Large overlap → SVAMP-tuned score isolates roughly the same samples as transfer; the mechanism failure is unlikely to be score-portability. Small overlap → SVAMP-tuned picks structurally different samples; retry behavior could differ.

## Use — GPU retry on flagged subset

```
cd /home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval

PYTHON_BIN=/home/work/GFlowPO/anaconda3/envs/prophet/bin/python \
SUBSET_INDICES_FILE=/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/svamp_tuned_retry_subset_q25_20260515.txt \
RUN_NAME=20260515_svamp_retry_exp_svamptuned_q25 \
TASK=svamp \
VOTE_METHOD=exp \
bash scripts/run_retry_policy_experiment.sh 1
```

Then generate a budget-matched complement-only random control with `select_svamp_tuned_random_retry_subset.py` (companion script) using the SVAMP-tuned manifest's `tau` and `n_flagged`.

## First 10 flagged test samples

- idx `15` | score `0.5576` | base `21.0` | prob `21.0` | correct `0`
- idx `17` | score `0.8392` | base `177.0` | prob `177.0` | correct `0`
- idx `71` | score `0.8685` | base `495.0` | prob `495.0` | correct `0`
- idx `79` | score `0.8642` | base `1891.0` | prob `1891.0` | correct `1`
- idx `111` | score `0.8176` | base `4.0` | prob `4.0` | correct `1`
- idx `135` | score `0.6285` | base `8.0` | prob `8.0` | correct `1`
- idx `142` | score `0.7672` | base `105.0` | prob `105.0` | correct `1`
- idx `147` | score `0.5179` | base `8066.0` | prob `8066.0` | correct `1`
- idx `193` | score `0.8441` | base `448.0` | prob `448.0` | correct `0`
- idx `197` | score `0.6438` | base `24.0` | prob `24.0` | correct `1`
