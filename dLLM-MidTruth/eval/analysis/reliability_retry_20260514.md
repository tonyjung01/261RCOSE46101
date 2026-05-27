# Reliability Retry Sweep — 20260514

Offline E4 simulation: retry only the least reliable samples and swap their baseline `exp_only` answer with an answer from an already existing artifact/source.

## Base accuracy

- GSM8K val:  70.62%
- GSM8K test: 67.42%
- SVAMP full: 86.33%

## Standalone retry-source accuracy

| Source | GSM8K test | SVAMP |
|---|---:|---:|
| cgap_vote | 68.94% | 86.33% |
| final_answer | 66.67% | 84.67% |
| prob_vote | 70.08% | 86.67% |
| blockactive_vote | 67.80% | 84.67% |
| retry_majority | 68.94% | 86.33% |
| oracle_pool | 70.45% | 87.33% |

## combined_score — val-selected retry budget

| Retry source | Best retry frac (val) | Val acc | Test acc | Test delta | SVAMP acc | SVAMP delta |
|---|---:|---:|---:|---:|---:|---:|
| cgap_vote | 5% | 70.52% | 68.18% | +0.76% | 86.33% | +0.00% |
| final_answer | 5% | 70.62% | 67.42% | +0.00% | 86.33% | +0.00% |
| prob_vote | 20% | 70.62% | 68.94% | +1.52% | 86.67% | +0.33% |
| blockactive_vote | 5% | 70.43% | 67.80% | +0.38% | 86.00% | -0.33% |
| retry_majority | 5% | 70.43% | 68.18% | +0.76% | 86.33% | +0.00% |

## logistic_broad_score — val-selected retry budget

| Retry source | Best retry frac (val) | Val acc | Test acc | Test delta | SVAMP acc | SVAMP delta |
|---|---:|---:|---:|---:|---:|---:|
| cgap_vote | 5% | 70.52% | 67.80% | +0.38% | 86.33% | +0.00% |
| final_answer | 5% | 70.62% | 67.42% | +0.00% | 86.33% | +0.00% |
| prob_vote | 25% | 70.71% | 68.94% | +1.52% | 86.67% | +0.33% |
| blockactive_vote | 5% | 70.33% | 67.80% | +0.38% | 86.00% | -0.33% |
| retry_majority | 5% | 70.43% | 67.80% | +0.38% | 86.33% | +0.00% |

## Notes

- This is not a fresh retry generation; it is an offline substitution test using existing answer sources.
- The baseline answer is reconstructed `exp_only` from the current debug artifact, matching Phase E/E2 labels.
- A positive result would mean the reliability score is not only abstention-useful, but also can target samples that benefit from switching answer sources.
