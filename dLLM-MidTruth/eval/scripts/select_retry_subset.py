"""Prepare the selective-retry subset from a base artifact.

This script operationalizes the current best reliability rule:

    if logistic_broad_score < tau:
        retry this sample

It reuses the Phase E feature / score-fitting logic so the retry subset is
defined on the same reliability space as the abstention and fallback analyses.
The output includes:

- a JSON manifest with score metadata
- a plain-text index list usable by `eval.py --subset_indices_file`
- a short Markdown summary for bookkeeping
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import analyze_reliability_retry as e4


DEFAULT_OUT_DIR = e4.DEFAULT_OUT_DIR
DEFAULT_SCORE_KEY = "logistic_broad_score"
DEFAULT_TAU = 0.5845


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


def _select_rows(rows, score_key, tau):
    selected = [r for r in rows if float(r[score_key]) <= tau]
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-artifact", default=e4.DEFAULT_GSM8K_BASE)
    parser.add_argument("--score-key", choices=["combined_score", "logistic_broad_score"], default=DEFAULT_SCORE_KEY)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--subset-split", choices=["val", "test", "full"], default="test")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--output-stem", default="")
    args = parser.parse_args()

    rows = _score_rows(args.base_artifact, args.subset_split)
    selected = _select_rows(rows, args.score_key, args.tau)

    date_str = datetime.now().strftime("%Y%m%d")
    stem = args.output_stem or f"retry_subset_{args.score_key}_{date_str}"
    os.makedirs(args.out_dir, exist_ok=True)

    manifest = {
        "base_artifact": args.base_artifact,
        "score_key": args.score_key,
        "tau": args.tau,
        "subset_split": args.subset_split,
        "n_total": len(rows),
        "n_selected": len(selected),
        "retry_rate": len(selected) / len(rows) if rows else 0.0,
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
    lines.append(f"# Retry Subset Manifest — {date_str}")
    lines.append("")
    lines.append(f"- base artifact: `{args.base_artifact}`")
    lines.append(f"- score key: `{args.score_key}`")
    lines.append(f"- tau: `{args.tau:.4f}`")
    lines.append(f"- subset split: `{args.subset_split}`")
    lines.append(f"- selected: `{manifest['n_selected']}/{manifest['n_total']}` (`{manifest['retry_rate']:.2%}`)")
    lines.append(f"- json: `{json_path}`")
    lines.append(f"- txt: `{txt_path}`")
    lines.append("")
    lines.append("## First 10 selected samples")
    lines.append("")
    for row in manifest["rows"][:10]:
        lines.append(
            f"- idx `{row['sample_index']}` | score `{row['score']:.4f}` | "
            f"base `{row['exp_only_answer']}` | prob `{row['prob_vote']}` | "
            f"correct `{row['is_correct']}`"
        )

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote: {json_path}")
    print(f"Wrote: {txt_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
