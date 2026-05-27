# GSM8K T=0 Selective Retry Budget Sweep — 20260515

Offline budget sweep on the existing T=0 retry artifact (`v6`). For each `k`, take the bottom-`k` flagged samples by `logistic_broad_score`, apply retry only on those, keep `exp_only` on the rest. No new GPU.

- retry artifact: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260514_gsm8k_retry_exp_testsubset_tau05845_v6/rank_0_generations.json`
- flagged manifest: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/retry_subset_gsm8k_test_logistic_broad_tau05845_20260514.json`
- test n: `264`; flagged total: `60`
- baseline `exp_only` full-test acc: `67.42%`

## Budget sweep results

| k | k/test | subset base acc | full-test acc | delta | fixes | hurts | net | fix/hurt ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 1.89% | 0.00% | 67.80% | `+0.38pt` | 1 | 0 | +1 | ∞ |
| 10 | 3.79% | 10.00% | 67.80% | `+0.38pt` | 2 | 1 | +1 | 2.00 |
| 15 | 5.68% | 13.33% | 67.80% | `+0.38pt` | 2 | 1 | +1 | 2.00 |
| 20 | 7.58% | 10.00% | 68.18% | `+0.76pt` | 3 | 1 | +2 | 3.00 |
| 25 | 9.47% | 16.00% | 68.56% | `+1.14pt` | 4 | 1 | +3 | 4.00 |
| 30 | 11.36% | 20.00% | 68.56% | `+1.14pt` | 4 | 1 | +3 | 4.00 |
| 40 | 15.15% | 25.00% | 68.94% | `+1.52pt` | 5 | 1 | +4 | 5.00 |
| 50 | 18.94% | 28.00% | 69.32% | `+1.89pt` | 6 | 1 | +5 | 6.00 |
| 60 | 22.73% | 30.00% | 70.08% | `+2.65pt` | 8 | 1 | +7 | 8.00 |

## Reading guide

- if `net` is flat or peaks early (e.g. at `k=10` or `20`), then most of the GSM8K selective retry gain concentrates at the very lowest-score tail; the original `tau=0.5845` budget (`k=60`) is wider than necessary.
- if `net` keeps growing with `k`, the budget choice is well-tuned and smaller budgets leave gain on the table.
- if `fix/hurt ratio` drops sharply with `k`, wider budgets dilute targeting (analogous to SVAMP q25 vs q40 pattern).
