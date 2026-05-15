"""SVAMP-tuned random-budget retry control subset.

Companion to `select_svamp_tuned_retry_subset.py`. Loads the SVAMP-tuned
score manifest, identifies the SVAMP test split, then draws a
budget-matched random subset from the SVAMP test unflagged complement
(or from the full test pool with `--no-exclude-flagged`).
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

import analyze_reliability as e1  # noqa: E402
import analyze_reliability_retry as e4  # noqa: E402
import select_svamp_tuned_retry_subset as sel  # noqa: E402


DEFAULT_OUT_DIR = e4.DEFAULT_OUT_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selective-manifest", required=True,
                        help="path to the SVAMP-tuned selective manifest json (used for tau and budget)")
    parser.add_argument("--k", type=int, default=None,
                        help="random subset size; defaults to the selective manifest's n_flagged_test")
    parser.add_argument("--random-seed", type=int, default=124)
    parser.add_argument("--exclude-flagged", action="store_true", default=True,
                        help="draw from the test unflagged complement only (default True)")
    parser.add_argument("--no-exclude-flagged", dest="exclude_flagged", action="store_false")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--output-stem", default="")
    args = parser.parse_args()

    with open(args.selective_manifest) as f:
        sel_manifest = json.load(f)
    tau = sel_manifest["tau"]
    quantile = sel_manifest["quantile"]
    k = args.k or sel_manifest["n_flagged_test"]

    print(f"[load] SVAMP merged + outer split (seed=42, frac=0.8)")
    svamp = sel._load_svamp_merged()
    val, test = sel._split_svamp(svamp)
    print(f"  test n={len(test)}")

    print(f"[fit] SVAMP-tuned score on SVAMP val (recover the same score used for selective)")
    score = sel._fit_score_on_svamp_val(val)
    test_scored = sel._apply_score(score, test)

    flagged_indices = {r["sample_index"] for r in test_scored if r["logistic_broad_score"] <= tau}
    if args.exclude_flagged:
        pool = [r for r in test_scored if r["sample_index"] not in flagged_indices]
    else:
        pool = test_scored

    if k > len(pool):
        raise ValueError(f"k={k} > pool size {len(pool)}")
    rng = random.Random(args.random_seed)
    selected = sorted(rng.sample(pool, k), key=lambda r: r["sample_index"])

    overlap = sum(1 for r in selected if r["sample_index"] in flagged_indices)
    base_correct = sum(r["is_correct"] for r in selected)
    base_acc = base_correct / len(selected) if selected else 0.0

    date_str = datetime.now().strftime("%Y%m%d")
    suffix = "complement" if args.exclude_flagged else "pure"
    stem = args.output_stem or f"svamp_tuned_random_retry_subset_{suffix}_seed{args.random_seed}_q{int(quantile*100):02d}_{date_str}"
    os.makedirs(args.out_dir, exist_ok=True)

    manifest = {
        "base_artifact": e4.DEFAULT_SVAMP_BASE,
        "score_fit_source": "SVAMP val (seed=42 outer split, frac=0.8)",
        "tau": tau,
        "quantile": quantile,
        "k": k,
        "random_seed": args.random_seed,
        "exclude_flagged": args.exclude_flagged,
        "n_test": len(test),
        "n_pool": len(pool),
        "n_selected": len(selected),
        "overlap_with_flagged": overlap,
        "random_subset_base_acc": base_acc,
        "selective_manifest": args.selective_manifest,
        "sample_indices": [int(r["sample_index"]) for r in selected],
        "rows": [
            {
                "sample_index": int(r["sample_index"]),
                "question": r["question"],
                "ground_truth": r["ground_truth"],
                "score": float(r["logistic_broad_score"]),
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
    lines.append(f"# SVAMP-tuned Random Retry Control Subset — {date_str}")
    lines.append("")
    lines.append(f"- subset type: **{suffix} random** (overlap with SVAMP-tuned flagged = `{overlap}/{len(selected)}`)")
    lines.append(f"- score fit source: SVAMP val (seed=42 outer, 80/20)")
    lines.append(f"- tau (from selective manifest): `{tau:.4f}` (val {quantile*100:.0f}%-quantile)")
    lines.append(f"- k: `{k}`")
    lines.append(f"- random seed: `{args.random_seed}`")
    lines.append(f"- SVAMP test n: `{len(test)}`  pool size: `{len(pool)}`")
    lines.append(f"- random subset base acc: `{base_acc*100:.2f}%`")
    lines.append("")
    lines.append("## Use — GPU random retry on this subset")
    lines.append("")
    lines.append("```")
    lines.append("cd /home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval")
    lines.append("")
    lines.append("PYTHON_BIN=/home/work/GFlowPO/anaconda3/envs/prophet/bin/python \\")
    lines.append(f"SUBSET_INDICES_FILE={txt_path} \\")
    lines.append(f"RUN_NAME=20260515_svamp_retry_exp_svamptuned_{suffix}_seed{args.random_seed} \\")
    lines.append("TASK=svamp \\")
    lines.append("VOTE_METHOD=exp \\")
    lines.append("bash scripts/run_retry_policy_experiment.sh 1")
    lines.append("```")

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote: {json_path}")
    print(f"Wrote: {txt_path}")
    print(f"Wrote: {md_path}")
    print()
    print(f"random subset base acc: {base_acc*100:.2f}%  (overlap with flagged: {overlap})")


if __name__ == "__main__":
    main()
