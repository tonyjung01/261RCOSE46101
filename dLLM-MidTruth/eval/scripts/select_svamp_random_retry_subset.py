"""SVAMP random-budget retry-control subset.

Companion to `select_svamp_retry_subset.py`. Draws a budget-matched
random subset from SVAMP, optionally restricted to the score-unflagged
complement (`--exclude-flagged`) so overlap with the score-flagged set is 0.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import analyze_reliability_retry as e4  # noqa: E402
import select_svamp_retry_subset as svamp_sel  # noqa: E402


DEFAULT_OUT_DIR = e4.DEFAULT_OUT_DIR
DEFAULT_K = 49  # match the SVAMP selective retry budget at tau=0.5845
DEFAULT_RANDOM_SEED = 124
DEFAULT_SCORE_KEY = "logistic_broad_score"
DEFAULT_TAU = 0.5845


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--exclude-flagged", action="store_true",
                        help="sample only from the score-unflagged complement (overlap with flagged = 0)")
    parser.add_argument("--score-key", choices=["combined_score", "logistic_broad_score"], default=DEFAULT_SCORE_KEY)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--output-stem", default="")
    args = parser.parse_args()

    score = svamp_sel._fit_score_on_gsm_val()
    svamp = svamp_sel._load_svamp_scored(score)
    flagged_indices = {r["sample_index"] for r in svamp if float(r[args.score_key]) <= args.tau}
    if args.exclude_flagged:
        pool = [r for r in svamp if r["sample_index"] not in flagged_indices]
    else:
        pool = svamp
    if args.k > len(pool):
        raise ValueError(f"k={args.k} > pool size {len(pool)}")
    rng = random.Random(args.random_seed)
    selected = sorted(rng.sample(pool, args.k), key=lambda r: r["sample_index"])

    overlap = sum(1 for r in selected if r["sample_index"] in flagged_indices)
    base_correct = sum(r["is_correct"] for r in selected)

    date_str = datetime.now().strftime("%Y%m%d")
    suffix = "complement" if args.exclude_flagged else "pure"
    stem = args.output_stem or f"svamp_random_retry_subset_{suffix}_seed{args.random_seed}_{date_str}"
    os.makedirs(args.out_dir, exist_ok=True)

    manifest = {
        "base_artifact": e4.DEFAULT_SVAMP_BASE,
        "k": args.k,
        "random_seed": args.random_seed,
        "exclude_flagged": args.exclude_flagged,
        "score_key": args.score_key,
        "score_tau": args.tau,
        "score_fit_source": "GSM8K val (seed=42, frac=0.8)",
        "n_total_svamp": len(svamp),
        "n_total_pool": len(pool),
        "n_selected": len(selected),
        "overlap_with_flagged": overlap,
        "random_subset_base_acc": base_correct / len(selected) if selected else 0,
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
                "score_flagged": r["sample_index"] in flagged_indices,
            }
            for r in selected
        ],
    }

    json_path = os.path.join(args.out_dir, f"{stem}.json")
    txt_path = os.path.join(args.out_dir, f"{stem}.txt")
    md_path = os.path.join(args.out_dir, f"{stem}.md")

    with open(json_path, "w") as f:
        json.dump(manifest, f, indent=2)
    with open(txt_path, "w") as f:
        for idx in manifest["sample_indices"]:
            f.write(f"{idx}\n")

    lines = []
    lines.append(f"# SVAMP Random Retry Control Subset — {date_str}")
    lines.append("")
    lines.append(f"- subset type: **{'complement-only' if args.exclude_flagged else 'pure random'}** (overlap with flagged = `{overlap}/{len(selected)}`)")
    lines.append(f"- k: `{args.k}` (matches the SVAMP selective retry budget at `tau={args.tau}`)")
    lines.append(f"- random seed: `{args.random_seed}`")
    lines.append(f"- pool size: `{len(pool)}`{' (unflagged only)' if args.exclude_flagged else ' (full SVAMP)'}")
    lines.append(f"- random subset base acc: `{manifest['random_subset_base_acc']*100:.2f}%` (SVAMP overall is `86.33%`, flagged is `55.10%`)")
    lines.append(f"- json: `{json_path}`")
    lines.append(f"- txt:  `{txt_path}`")
    lines.append("")
    lines.append("## Use — GPU random retry on this subset")
    lines.append("")
    lines.append("```")
    lines.append("cd /home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval")
    lines.append("")
    lines.append("PYTHON_BIN=/home/work/GFlowPO/anaconda3/envs/prophet/bin/python \\")
    lines.append(f"SUBSET_INDICES_FILE={txt_path} \\")
    lines.append(f"RUN_NAME=20260515_svamp_retry_exp_{suffix}_seed{args.random_seed} \\")
    lines.append("TASK=svamp \\")
    lines.append("VOTE_METHOD=exp \\")
    lines.append("bash scripts/run_retry_policy_experiment.sh 1")
    lines.append("```")
    lines.append("")
    lines.append("## First 10 selected SVAMP samples")
    lines.append("")
    for row in manifest["rows"][:10]:
        f_str = "Y" if row["score_flagged"] else "N"
        lines.append(
            f"- idx `{row['sample_index']}` | score `{row['score']:.4f}` (flagged `{f_str}`) | "
            f"base `{row['exp_only_answer']}` | correct `{row['is_correct']}`"
        )

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote: {json_path}")
    print(f"Wrote: {txt_path}")
    print(f"Wrote: {md_path}")
    print()
    print(f"random subset base acc: {manifest['random_subset_base_acc']*100:.2f}%  (flagged 55.10% / overall 86.33%)")
    print(f"overlap with score-flagged set: {overlap}")


if __name__ == "__main__":
    main()
