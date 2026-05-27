# Math500 Reliability-aware Abstention — 20260514

Source: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260430_cgap_answer_window5_prob_mean_rawsum_bs4_all_debug/math500_gen128_steps64_vote_confidence_gap_answer_window5_prob_mean_rawsum_bs4/rank_0_generations.json`
outer split seed=42; val_frac=0.8; inner_seed=7; inner_frac=0.8

Math500 n=500 (val 400 / test 100)

Base accuracy (no abstention):
- Math500 full: 24.40%
- Math500 val:  25.25%
- Math500 test: 21.00%

## Single-feature val Pearson

| Feature | Val Pearson |
|---|---:|
| last_change_step_frac | -0.267 |
| n_valid | +0.262 |
| valid_ratio | +0.262 |
| bucket_a_ratio | -0.256 |
| awnf_ratio | -0.256 |
| first_valid_step_frac | -0.191 |
| cat_no_boxed_no_answer_tag_ratio | -0.184 |
| max_gap_step_frac | +0.178 |
| cat_boxedboxed_ratio | -0.176 |
| max_gap | +0.152 |
| cat_no_boxed_but_answer_tag_ratio | -0.105 |
| mean_gap | +0.096 |
| std_gap | +0.090 |
| parse_failed_ratio | -0.086 |
| n_unique_answers | -0.076 |
| cat_boxed_like_ratio | -0.066 |
| last_change_gap | -0.050 |
| bucket_b_ratio | -0.027 |
| first_valid_gap | +0.005 |

Z-score combined features (kept |corr|>=0.05): ['max_gap', 'mean_gap', 'last_change_gap', 'std_gap', 'n_valid', 'n_unique_answers', 'last_change_step_frac', 'max_gap_step_frac', 'first_valid_step_frac', 'valid_ratio', 'awnf_ratio', 'parse_failed_ratio', 'bucket_a_ratio', 'cat_boxedboxed_ratio', 'cat_boxed_like_ratio', 'cat_no_boxed_no_answer_tag_ratio', 'cat_no_boxed_but_answer_tag_ratio']

## Math500 Validation

| Score | AURC | SelAcc@80% | SelAcc@50% | Cov@90%acc |
|---|---:|---:|---:|---:|
| zsum_combined | 0.6975 | 29.38% | 36.50% | n/a |
| best_single (n_valid) | 0.7013 | 29.69% | 36.00% | n/a |
| logistic_broad | 0.6972 | 30.31% | 37.50% | n/a |
| oracle | 0.6628 | 31.56% | 50.50% | — |
| random | 0.7442 | 25.62% | 24.00% | — |

## Math500 Test (one-shot)

| Score | AURC | SelAcc@80% | SelAcc@50% | Cov@90%acc |
|---|---:|---:|---:|---:|
| zsum_combined | 0.7560 | 22.50% | 32.00% | n/a |
| best_single (n_valid) | 0.7628 | 22.50% | 28.00% | n/a |
| logistic_broad | 0.7568 | 22.50% | 30.00% | n/a |
| oracle | 0.7196 | 26.25% | 42.00% | — |
| random | 0.8254 | 18.75% | 14.00% | — |

## Notes

- Target label is offline `exp_only` correctness reconstructed from the valid events in this artifact, scored with `utils.parsers.is_equiv`.
- Feature set is Math500-specific: it includes AWNF / Bucket-A structure in addition to valid-event gap statistics.
- This should be read as a task-specific reliability pass, not as a direct continuation of the voting/hybrid line.
