"""GSM8K T=0 selective retry budget sweep (offline, no new GPU).

For each candidate budget `k`, take the bottom-`k` samples by
reliability score from the existing flagged 60, apply retry only on
those `k` samples (using the existing v6 T=0 retry artifact), keep
`exp_only` on the rest, evaluate full-test accuracy.

Question: is `k=60` (`22.73%` of test, our default tau=0.5845) the
sweet spot, or does a smaller budget capture most of the gain?

This is the GSM8K analog to the SVAMP q25 vs q40 budget probe. T=0
framework; no new GPU runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import analyze_reliability as e1  # noqa: E402
import analyze_reliability_retry as e4  # noqa: E402
import evaluate_true_retry as e_eval  # noqa: E402


DEFAULT_OUT_DIR = e4.DEFAULT_OUT_DIR
DEFAULT_RETRY_ARTIFACT = os.path.join(
    REPO_ROOT,
    "outputs/LLaDA-8B-Instruct/20260514_gsm8k_retry_exp_testsubset_tau05845_v6/rank_0_generations.json",
)
DEFAULT_SELECTIVE_MANIFEST = os.path.join(
    REPO_ROOT,
    "analysis/retry_subset_gsm8k_test_logistic_broad_tau05845_20260514.json",
)
BUDGET_CANDIDATES = [5, 10, 15, 20, 25, 30, 40, 50, 60]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-artifact", default=e4.DEFAULT_GSM8K_BASE)
    parser.add_argument("--prob-artifact", default=e4.DEFAULT_GSM8K_PROB)
    parser.add_argument("--block-artifact", default=e4.DEFAULT_GSM8K_BLOCKACTIVE)
    parser.add_argument("--retry-artifact", default=DEFAULT_RETRY_ARTIFACT)
    parser.add_argument("--selective-manifest", default=DEFAULT_SELECTIVE_MANIFEST)
    parser.add_argument("--budgets", nargs="+", type=int, default=BUDGET_CANDIDATES)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--output-stem", default="")
    args = parser.parse_args()

    print(f"[load] GSM8K test rows")
    test_rows = e_eval._score_rows(args.base_artifact, args.prob_artifact, args.block_artifact, "test")
    print(f"  test n={len(test_rows)}")

    print(f"[load] flagged manifest")
    with open(args.selective_manifest) as f:
        manifest = json.load(f)
    flagged_rows = sorted(manifest["rows"], key=lambda r: r["score"])
    print(f"  flagged total: {len(flagged_rows)}")

    # Retry artifact by question
    retry_map = e_eval._load_retry_answers(args.retry_artifact)
    print(f"  retry artifact: {os.path.basename(os.path.dirname(args.retry_artifact))}")

    # Base test correctness
    base_correct = sum(int(e1._is_correct(r["exp_only_answer"], r["ground_truth"])) for r in test_rows)
    base_acc = base_correct / len(test_rows)
    print(f"  base test acc = {base_acc*100:.2f}% ({base_correct}/{len(test_rows)})")

    rows_by_idx = {r["sample_index"]: r for r in test_rows}

    sweep = []
    print(f"\n[budget sweep]")
    print(f"{'k':>4} {'flagged_subset_base_acc':>26} {'full_test_acc':>14} {'delta':>9} {'fixes':>6} {'hurts':>6}")
    for k in args.budgets:
        if k > len(flagged_rows):
            continue
        bottom_k = flagged_rows[:k]
        flagged_questions = set()
        for row in bottom_k:
            full_row = rows_by_idx[row["sample_index"]]
            flagged_questions.add(full_row["question"])

        correct = 0
        fixes = 0
        hurts = 0
        for r in test_rows:
            gt = r["ground_truth"]
            base = r["exp_only_answer"]
            if r["question"] in flagged_questions:
                retry_row = retry_map.get(r["question"])
                use = retry_row["exp_only_answer"] if (retry_row and retry_row.get("exp_only_answer") is not None) else base
            else:
                use = base
            is_corr = int(e1._is_correct(use, gt))
            base_corr = int(e1._is_correct(base, gt))
            correct += is_corr
            if r["question"] in flagged_questions:
                if is_corr and not base_corr:
                    fixes += 1
                if (not is_corr) and base_corr:
                    hurts += 1
        acc = correct / len(test_rows)
        delta = acc - base_acc
        # subset-local base acc (avg over the k flagged samples)
        subset_base_correct = sum(int(e1._is_correct(rows_by_idx[r["sample_index"]]["exp_only_answer"], rows_by_idx[r["sample_index"]]["ground_truth"])) for r in bottom_k)
        subset_base_acc = subset_base_correct / k
        sweep.append({
            "k": k,
            "k_frac": k / len(test_rows),
            "subset_base_acc": subset_base_acc,
            "full_test_acc": acc,
            "delta_vs_base": delta,
            "fixes": fixes,
            "hurts": hurts,
            "net": fixes - hurts,
        })
        print(f"{k:>4d} {subset_base_acc*100:>25.2f}% {acc*100:>13.2f}% {delta*100:>+8.2f}pt {fixes:>6d} {hurts:>6d}")

    # === Output ===
    os.makedirs(args.out_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    stem = args.output_stem or f"gsm8k_retry_budget_sweep_{date_str}"

    out = {
        "config": {
            "retry_artifact": args.retry_artifact,
            "selective_manifest": args.selective_manifest,
            "budgets": args.budgets,
        },
        "base_acc": base_acc,
        "sweep": sweep,
    }
    json_path = os.path.join(args.out_dir, f"{stem}.json")
    md_path = os.path.join(args.out_dir, f"{stem}.md")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote: {json_path}")

    lines = []
    lines.append(f"# GSM8K T=0 Selective Retry Budget Sweep — {date_str}")
    lines.append("")
    lines.append("Offline budget sweep on the existing T=0 retry artifact (`v6`). For each `k`, take the bottom-`k` flagged samples by `logistic_broad_score`, apply retry only on those, keep `exp_only` on the rest. No new GPU.")
    lines.append("")
    lines.append(f"- retry artifact: `{args.retry_artifact}`")
    lines.append(f"- flagged manifest: `{args.selective_manifest}`")
    lines.append(f"- test n: `{len(test_rows)}`; flagged total: `{len(flagged_rows)}`")
    lines.append(f"- baseline `exp_only` full-test acc: `{base_acc*100:.2f}%`")
    lines.append("")
    lines.append("## Budget sweep results")
    lines.append("")
    lines.append("| k | k/test | subset base acc | full-test acc | delta | fixes | hurts | net | fix/hurt ratio |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in sweep:
        ratio = f"{row['fixes'] / row['hurts']:.2f}" if row['hurts'] > 0 else "∞"
        lines.append(
            f"| {row['k']} | {row['k_frac']*100:.2f}% | {row['subset_base_acc']*100:.2f}% "
            f"| {row['full_test_acc']*100:.2f}% | `{row['delta_vs_base']*100:+.2f}pt` "
            f"| {row['fixes']} | {row['hurts']} | {row['net']:+d} | {ratio} |"
        )
    lines.append("")
    lines.append("## Reading guide")
    lines.append("")
    lines.append("- if `net` is flat or peaks early (e.g. at `k=10` or `20`), then most of the GSM8K selective retry gain concentrates at the very lowest-score tail; the original `tau=0.5845` budget (`k=60`) is wider than necessary.")
    lines.append("- if `net` keeps growing with `k`, the budget choice is well-tuned and smaller budgets leave gain on the table.")
    lines.append("- if `fix/hurt ratio` drops sharply with `k`, wider budgets dilute targeting (analogous to SVAMP q25 vs q40 pattern).")
    lines.append("")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
