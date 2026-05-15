"""Prepare a random-budget retry control subset.

This is the control for Phase E-Retry-Control. Instead of selecting the
bottom-k samples by reliability score (the E-Retry policy), we pick a
random k samples from the same test split with a different seed. Same
budget, same evaluation, no score targeting.

Comparing the two retry artifacts (score-targeted vs random) measures
the marginal value of the reliability score for retry budget allocation,
separately from "retry helps in general."

Outputs match `select_retry_subset.py` (manifest json, sample_indices txt,
short md), so the GPU rerun pipeline (`eval.py --subset_indices_file`) is
the same.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime

import analyze_reliability_retry as e4


DEFAULT_OUT_DIR = e4.DEFAULT_OUT_DIR
DEFAULT_RANDOM_SEED = 123
DEFAULT_K = 60  # match the selective retry budget on test split (22.73%)
DEFAULT_SCORE_KEY = "logistic_broad_score"


def _score_rows(base_artifact, subset_split):
    feat_rows = e4._load_feature_rows(base_artifact)
    base_answers = e4._load_answer_artifact(base_artifact)
    merged = e4._merge_sources(feat_rows, base_answers, base_answers, base_answers)
    split = e4._fit_scores(merged)
    if subset_split == "val":
        rows = split["val"]
    elif subset_split == "test":
        rows = split["test"]
    elif subset_split == "full":
        rows = split["val"] + split["test"]
    else:
        raise ValueError(f"Unsupported subset_split: {subset_split}")
    rows = sorted(rows, key=lambda r: r["sample_index"])
    return rows


def _random_select(rows, k, seed, exclude_indices=None):
    """Uniform random selection of k rows, optionally excluding a flagged set.

    We allow `exclude_indices=None` (true random control), as well as
    `exclude_indices=<flagged_set>` (random sampled from the complement,
    which is a stricter control: no overlap with the score-flagged set
    so the targeting effect is not contaminated).
    """
    rng = random.Random(seed)
    pool = rows if not exclude_indices else [r for r in rows if r["sample_index"] not in exclude_indices]
    if k > len(pool):
        raise ValueError(f"k={k} > pool size {len(pool)}")
    return sorted(rng.sample(pool, k), key=lambda r: r["sample_index"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-artifact", default=e4.DEFAULT_GSM8K_BASE)
    parser.add_argument("--subset-split", choices=["val", "test", "full"], default="test")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="number of samples to retry (match the selective retry budget)")
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--exclude-flagged", action="store_true",
                        help="exclude samples already flagged by the score (sample from complement only); cleaner control")
    parser.add_argument("--score-key", choices=["combined_score", "logistic_broad_score"], default=DEFAULT_SCORE_KEY)
    parser.add_argument("--score-tau", type=float, default=0.5845)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--output-stem", default="")
    args = parser.parse_args()

    rows = _score_rows(args.base_artifact, args.subset_split)
    exclude = None
    if args.exclude_flagged:
        exclude = {r["sample_index"] for r in rows if float(r[args.score_key]) <= args.score_tau}
    selected = _random_select(rows, args.k, args.random_seed, exclude)

    date_str = datetime.now().strftime("%Y%m%d")
    stem = args.output_stem or f"random_retry_subset_{args.subset_split}_seed{args.random_seed}_{date_str}"
    os.makedirs(args.out_dir, exist_ok=True)

    manifest = {
        "base_artifact": args.base_artifact,
        "subset_split": args.subset_split,
        "k": args.k,
        "random_seed": args.random_seed,
        "exclude_flagged": args.exclude_flagged,
        "score_key": args.score_key,
        "score_tau": args.score_tau,
        "n_total_pool": len(rows) - (len(exclude) if exclude else 0),
        "n_total_split": len(rows),
        "n_selected": len(selected),
        "sample_indices": [int(r["sample_index"]) for r in selected],
        "rows": [
            {
                "sample_index": int(r["sample_index"]),
                "question": r["question"],
                "ground_truth": r["ground_truth"],
                "score": float(r[args.score_key]),
                "exp_only_answer": r["exp_only_answer"],
                "prob_vote": r.get("prob_vote"),
                "is_correct": int(r["is_correct"]),
                "score_flagged": float(r[args.score_key]) <= args.score_tau,
            }
            for r in selected
        ],
    }

    # Quick descriptive stats on the random subset for sanity
    base_correct = sum(r["is_correct"] for r in manifest["rows"])
    flagged_overlap = sum(1 for r in manifest["rows"] if r["score_flagged"])
    manifest["random_subset_base_acc"] = base_correct / len(selected) if selected else 0.0
    manifest["overlap_with_flagged"] = flagged_overlap

    json_path = os.path.join(args.out_dir, f"{stem}.json")
    txt_path = os.path.join(args.out_dir, f"{stem}.txt")
    md_path = os.path.join(args.out_dir, f"{stem}.md")

    with open(json_path, "w") as f:
        json.dump(manifest, f, indent=2)

    with open(txt_path, "w") as f:
        for idx in manifest["sample_indices"]:
            f.write(f"{idx}\n")

    lines = []
    lines.append(f"# Random Retry Control Subset — {date_str}")
    lines.append("")
    lines.append(f"- base artifact: `{args.base_artifact}`")
    lines.append(f"- subset split: `{args.subset_split}` (n={manifest['n_total_split']})")
    lines.append(f"- k: `{args.k}` (random retry budget)")
    lines.append(f"- random seed: `{args.random_seed}`")
    lines.append(f"- exclude flagged: `{args.exclude_flagged}` (score key `{args.score_key}`, tau `{args.score_tau}`)")
    lines.append(f"- random pool size: `{manifest['n_total_pool']}`")
    lines.append(f"- random subset base acc: `{manifest['random_subset_base_acc']*100:.2f}%` (`{base_correct}/{len(selected)}`)")
    lines.append(f"- overlap with score-flagged set: `{flagged_overlap}`")
    lines.append(f"- json: `{json_path}`")
    lines.append(f"- txt: `{txt_path}`")
    lines.append("")
    lines.append("## Use")
    lines.append("")
    lines.append("Feed `txt` into the same retry pipeline used for the score-targeted retry. Recommended env-explicit form (matches the env used for the v6/v1 retry runs):")
    lines.append("")
    lines.append("```")
    lines.append("cd /home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval")
    lines.append("")
    lines.append("PYTHON_BIN=/home/work/GFlowPO/anaconda3/envs/prophet/bin/python \\")
    lines.append(f"SUBSET_INDICES_FILE={txt_path} \\")
    lines.append("RUN_NAME=20260514_gsm8k_retry_exp_random60_seed123 \\")
    lines.append("VOTE_METHOD=exp \\")
    lines.append("bash scripts/run_retry_policy_experiment.sh 1")
    lines.append("```")
    lines.append("")
    lines.append("Note: trailing `1` is the GPU index. Adjust if a different GPU is free.")
    lines.append("")
    lines.append("Then evaluate the resulting retry artifact head-to-head against the selective-retry artifact with `evaluate_retry_control.py`. Recommended env-explicit form:")
    lines.append("")
    lines.append("```")
    lines.append("/home/work/GFlowPO/anaconda3/envs/prophet/bin/python \\")
    lines.append("  /home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/scripts/evaluate_retry_control.py \\")
    lines.append("  --selective-retry-artifact /home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260514_gsm8k_retry_exp_testsubset_tau05845_v6/rank_0_generations.json \\")
    lines.append("  --selective-manifest /home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/retry_subset_gsm8k_test_logistic_broad_tau05845_20260514.json \\")
    lines.append("  --random-retry-artifact /home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260514_gsm8k_retry_exp_random60_seed123/rank_0_generations.json \\")
    lines.append(f"  --random-manifest {json_path} \\")
    lines.append("  --answer-kind exp_only")
    lines.append("```")
    lines.append("")
    lines.append("## Overlap caveat")
    lines.append("")
    lines.append(f"This is a pure random control: the random subset can overlap with the score-flagged set (here `{flagged_overlap}` of `{args.k}` samples). Pure random is the simplest budget-matched control; because it may overlap with the flagged set, a complement-only random control remains a stricter optional follow-up (`select_random_retry_subset.py --exclude-flagged`).")
    lines.append("")
    lines.append("## First 10 selected samples")
    lines.append("")
    for row in manifest["rows"][:10]:
        flag = "Y" if row["score_flagged"] else "N"
        lines.append(
            f"- idx `{row['sample_index']}` | score `{row['score']:.4f}` (flagged `{flag}`) | "
            f"base `{row['exp_only_answer']}` | prob `{row['prob_vote']}` | "
            f"correct `{row['is_correct']}`"
        )

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote: {json_path}")
    print(f"Wrote: {txt_path}")
    print(f"Wrote: {md_path}")
    print()
    print(f"random subset base acc: {manifest['random_subset_base_acc']*100:.2f}%  (vs flagged 30.00% / overall test 67.42%)")
    print(f"overlap with score-flagged set: {flagged_overlap}")


if __name__ == "__main__":
    main()
