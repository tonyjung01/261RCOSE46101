"""SVAMP-tuned reliability score for selective retry.

Mechanism question: the GSM8K-transferred score (`logistic_broad` fit on
GSM8K val) gave `+0.00pt` selective retry on SVAMP. Is the failure
because the score family is not portable, or because SVAMP's "difficulty"
just is not the retry-rescuable kind?

This script answers the first half: fit a SVAMP-native `logistic_broad`
on a SVAMP val split, apply to SVAMP test, flag the bottom 25%-quantile,
and emit a retry manifest. The retry itself runs as a separate GPU step.

Split: same 80/20 outer split (seed=42) as GSM8K, applied to SVAMP rows.
SVAMP n=300 → val=240, test=60. Bottom 25% of val sets `tau`. Test
flagged budget is small (typically ~15 samples). Single seed, small-n
test — we report it that way.

Pre-GPU offline diagnostics
- compare the SVAMP-tuned flagged-test set against the GSM8K-transferred
  flagged-test set (where they overlap, how many they differ on)
- check offline fix potential on the SVAMP-tuned flagged samples
  (`prob_vote` correct on samples where `exp_only` was wrong)
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
import analyze_reliability_v2 as e2  # noqa: E402


DEFAULT_OUT_DIR = e4.DEFAULT_OUT_DIR
DEFAULT_SCORE_KEY = "logistic_broad_score"


def _load_svamp_merged():
    feat = e4._load_feature_rows(e4.DEFAULT_SVAMP_BASE)
    base_ans = e4._load_answer_artifact(e4.DEFAULT_SVAMP_BASE)
    prob_ans = e4._load_answer_artifact(e4.DEFAULT_SVAMP_PROB)
    block_ans = e4._load_answer_artifact(e4.DEFAULT_SVAMP_BLOCKACTIVE)
    return e4._merge_sources(feat, base_ans, prob_ans, block_ans)


def _split_svamp(rows):
    """Same outer split as GSM8K (seed=42, val=0.8). Returns (val, test)."""
    return e1._split(rows)


def _fit_score_on_svamp_val(svamp_val):
    """Fit logistic_broad on SVAMP val (mirrors GSM8K's pipeline)."""
    zsum_spec = e1._fit_combined_score(svamp_val, e1.FEATURES)
    broad_features = list(zsum_spec.keys())
    inner_train, inner_dev = e2._inner_split(svamp_val)
    xt, yt, train_stats = e2._prepare_matrix(inner_train, broad_features)
    best = None
    for l2 in e2.L2_GRID:
        model = e2._fit_logreg_numpy(xt, yt, l2=l2)
        dev_scored = e2._score_rows(
            inner_dev, "model_score",
            e2._predict_scores(inner_dev, broad_features, train_stats, model),
        )
        aurc = e2._curve_pack(dev_scored, "model_score")["aurc"]
        if best is None or aurc < best["aurc"]:
            best = {"l2": l2, "aurc": aurc}
    x_full, y_full, full_stats = e2._prepare_matrix(svamp_val, broad_features)
    final_model = e2._fit_logreg_numpy(x_full, y_full, l2=best["l2"])
    return {
        "zsum_spec": zsum_spec,
        "broad_features": broad_features,
        "full_stats": full_stats,
        "final_model": final_model,
        "best_l2": best["l2"],
        "best_dev_aurc": best["aurc"],
    }


def _apply_score(score, rows):
    rows = [dict(r) for r in rows]
    e1._apply_combined_score(rows, score["zsum_spec"])
    s = e2._predict_scores(rows, score["broad_features"], score["full_stats"], score["final_model"])
    for r, sc in zip(rows, s):
        r["logistic_broad_score"] = float(sc)
    return rows


def _fit_gsm_score():
    """For comparison: the GSM8K-val-fitted score (the previous SVAMP cross-task)."""
    gsm_feat = e4._load_feature_rows(e4.DEFAULT_GSM8K_BASE)
    gsm_base_ans = e4._load_answer_artifact(e4.DEFAULT_GSM8K_BASE)
    gsm_prob_ans = e4._load_answer_artifact(e4.DEFAULT_GSM8K_PROB)
    gsm_block_ans = e4._load_answer_artifact(e4.DEFAULT_GSM8K_BLOCKACTIVE)
    gsm_merged = e4._merge_sources(gsm_feat, gsm_base_ans, gsm_prob_ans, gsm_block_ans)
    gsm_val, _ = e1._split(gsm_merged)
    return _fit_score_on_svamp_val(gsm_val)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantile", type=float, default=0.25,
                        help="bottom quantile of val score to set tau (default 0.25, matches GSM8K)")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--output-stem", default="")
    args = parser.parse_args()

    print("[load] SVAMP merged + outer split (seed=42, frac=0.8)")
    svamp = _load_svamp_merged()
    val, test = _split_svamp(svamp)
    print(f"  val n={len(val)}  test n={len(test)}")
    print(f"  val base acc:  {sum(r['is_correct'] for r in val)/len(val)*100:.2f}%")
    print(f"  test base acc: {sum(r['is_correct'] for r in test)/len(test)*100:.2f}%")

    print(f"\n[fit] SVAMP-tuned logistic_broad on SVAMP val")
    svamp_score = _fit_score_on_svamp_val(val)
    print(f"  best L2: {svamp_score['best_l2']}  inner-dev AURC: {svamp_score['best_dev_aurc']:.4f}")
    print(f"  features kept: {svamp_score['broad_features']}")

    val_scored = _apply_score(svamp_score, val)
    test_scored = _apply_score(svamp_score, test)

    # Determine tau on val
    val_scores = sorted(r["logistic_broad_score"] for r in val_scored)
    k_q = max(1, int(round(args.quantile * len(val_scores))))
    tau = val_scores[k_q - 1]
    print(f"\n[tau] val {args.quantile*100:.0f}%-quantile = {tau:.4f}")

    # Apply to test
    flagged_test = [r for r in test_scored if r["logistic_broad_score"] <= tau]
    unflagged_test = [r for r in test_scored if r["logistic_broad_score"] > tau]
    def acc(rows): return sum(r["is_correct"] for r in rows) / len(rows) if rows else 0
    print(f"  flagged on test:    {len(flagged_test)}/{len(test)} ({len(flagged_test)/len(test)*100:.2f}%)  base_acc={acc(flagged_test)*100:.2f}%")
    print(f"  unflagged on test:  {len(unflagged_test)}/{len(test)}  base_acc={acc(unflagged_test)*100:.2f}%")

    # Offline diagnostics
    print(f"\n[diagnostics] offline fix potential on SVAMP-tuned flagged test samples")
    base_wrong = [r for r in flagged_test if not r["is_correct"]]
    prob_can_fix = [r for r in base_wrong if r.get("prob_vote") is not None
                    and e1._is_correct(r["prob_vote"], r["ground_truth"])]
    print(f"  flagged base-wrong: {len(base_wrong)}/{len(flagged_test)}")
    print(f"  prob_vote correct on base-wrong: {len(prob_can_fix)}/{len(base_wrong)} (offline rescue ceiling)")

    # Compare against GSM8K-transferred flagged-test
    print(f"\n[diagnostics] comparison with GSM8K-transferred flagged set on SVAMP test")
    gsm_score = _fit_gsm_score()
    test_gsm_scored = _apply_score(gsm_score, test)
    # GSM8K-fitted tau was 0.5845 (its val 25%-quantile applied to all-SVAMP earlier);
    # apply the same value here for direct comparison
    gsm_tau = 0.5845
    gsm_flagged_test = {r["sample_index"] for r in test_gsm_scored if r["logistic_broad_score"] <= gsm_tau}
    svamp_flagged_test = {r["sample_index"] for r in flagged_test}
    overlap = svamp_flagged_test & gsm_flagged_test
    svamp_only = svamp_flagged_test - gsm_flagged_test
    gsm_only = gsm_flagged_test - svamp_flagged_test
    print(f"  SVAMP-tuned flagged on test:      {len(svamp_flagged_test)} samples")
    print(f"  GSM8K-transferred flagged on test: {len(gsm_flagged_test)} samples")
    print(f"  overlap:        {len(overlap)}")
    print(f"  SVAMP-only:     {len(svamp_only)}")
    print(f"  GSM8K-only:     {len(gsm_only)}")

    date_str = datetime.now().strftime("%Y%m%d")
    stem = args.output_stem or f"svamp_tuned_retry_subset_q{int(args.quantile*100):02d}_{date_str}"
    os.makedirs(args.out_dir, exist_ok=True)

    manifest = {
        "base_artifact": e4.DEFAULT_SVAMP_BASE,
        "score_key": DEFAULT_SCORE_KEY,
        "score_fit_source": "SVAMP val (seed=42 outer split, frac=0.8)",
        "tau": tau,
        "quantile": args.quantile,
        "best_l2": svamp_score["best_l2"],
        "best_dev_aurc": svamp_score["best_dev_aurc"],
        "outer_split_seed": e1.SEED,
        "outer_split_frac": e1.VALIDATION_FRAC,
        "n_val": len(val),
        "n_test": len(test),
        "val_base_acc": sum(r["is_correct"] for r in val) / len(val),
        "test_base_acc": sum(r["is_correct"] for r in test) / len(test),
        "n_flagged_test": len(flagged_test),
        "n_unflagged_test": len(unflagged_test),
        "flagged_base_acc": acc(flagged_test),
        "unflagged_base_acc": acc(unflagged_test),
        "offline_diagnostics": {
            "base_wrong_in_flagged": len(base_wrong),
            "prob_vote_can_fix": len(prob_can_fix),
        },
        "comparison_with_gsm8k_transferred": {
            "svamp_tuned_flagged_n": len(svamp_flagged_test),
            "gsm8k_transferred_flagged_n": len(gsm_flagged_test),
            "overlap": len(overlap),
            "svamp_only": len(svamp_only),
            "gsm_only": len(gsm_only),
        },
        "sample_indices": [int(r["sample_index"]) for r in flagged_test],
        "rows": [
            {
                "sample_index": int(r["sample_index"]),
                "question": r["question"],
                "ground_truth": r["ground_truth"],
                "score": float(r[DEFAULT_SCORE_KEY]),
                "exp_only_answer": r["exp_only_answer"],
                "prob_vote": r.get("prob_vote"),
                "is_correct": int(r["is_correct"]),
            }
            for r in flagged_test
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
    lines.append(f"# SVAMP-tuned Selective Retry Subset — {date_str}")
    lines.append("")
    lines.append("Mechanism check for the SVAMP mixed-transfer result. Score refit on SVAMP val (same 80/20 outer split, same logistic_broad pipeline); retry budget defined by val 25%-quantile of the new score; selective retry applied to held-out SVAMP test.")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- score fit source: SVAMP val ({len(val)} samples), val_base_acc `{manifest['val_base_acc']*100:.2f}%`")
    lines.append(f"- best L2: `{svamp_score['best_l2']}`, inner-dev AURC: `{svamp_score['best_dev_aurc']:.4f}`")
    lines.append(f"- features kept: `{svamp_score['broad_features']}`")
    lines.append(f"- tau: `{tau:.4f}` (val {args.quantile*100:.0f}%-quantile)")
    lines.append(f"- SVAMP test n: `{len(test)}`, test_base_acc `{manifest['test_base_acc']*100:.2f}%`")
    lines.append(f"- flagged on test: `{len(flagged_test)}/{len(test)}` ({len(flagged_test)/len(test)*100:.2f}%)  base_acc `{manifest['flagged_base_acc']*100:.2f}%`")
    lines.append(f"- unflagged on test: `{len(unflagged_test)}/{len(test)}`  base_acc `{manifest['unflagged_base_acc']*100:.2f}%`")
    lines.append("")
    lines.append("## Offline diagnostics on flagged test samples")
    lines.append("")
    lines.append(f"- base-wrong in flagged: `{len(base_wrong)}/{len(flagged_test)}`")
    lines.append(f"- offline rescue ceiling (`prob_vote` correct on base-wrong): `{len(prob_can_fix)}/{len(base_wrong)}`")
    lines.append("")
    lines.append("If the offline rescue ceiling is `0/N`, no static fallback can help and retry rescue depends entirely on generation variance. If it is `k > 0`, at least `k` of the flagged samples are recoverable by an existing offline source.")
    lines.append("")
    lines.append("## Comparison with the GSM8K-transferred flagged set (on the same SVAMP test split)")
    lines.append("")
    lines.append("| | n |")
    lines.append("|---|---:|")
    lines.append(f"| SVAMP-tuned flagged | {len(svamp_flagged_test)} |")
    lines.append(f"| GSM8K-transferred flagged (tau=0.5845) | {len(gsm_flagged_test)} |")
    lines.append(f"| overlap | {len(overlap)} |")
    lines.append(f"| SVAMP-only | {len(svamp_only)} |")
    lines.append(f"| GSM8K-only | {len(gsm_only)} |")
    lines.append("")
    lines.append("Large overlap → SVAMP-tuned score isolates roughly the same samples as transfer; the mechanism failure is unlikely to be score-portability. Small overlap → SVAMP-tuned picks structurally different samples; retry behavior could differ.")
    lines.append("")
    lines.append("## Use — GPU retry on flagged subset")
    lines.append("")
    lines.append("```")
    lines.append("cd /home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval")
    lines.append("")
    lines.append("PYTHON_BIN=/home/work/GFlowPO/anaconda3/envs/prophet/bin/python \\")
    lines.append(f"SUBSET_INDICES_FILE={txt_path} \\")
    lines.append(f"RUN_NAME=20260515_svamp_retry_exp_svamptuned_q{int(args.quantile*100):02d} \\")
    lines.append("TASK=svamp \\")
    lines.append("VOTE_METHOD=exp \\")
    lines.append("bash scripts/run_retry_policy_experiment.sh 1")
    lines.append("```")
    lines.append("")
    lines.append("Then generate a budget-matched complement-only random control with `select_svamp_tuned_random_retry_subset.py` (companion script) using the SVAMP-tuned manifest's `tau` and `n_flagged`.")
    lines.append("")
    lines.append("## First 10 flagged test samples")
    lines.append("")
    for row in manifest["rows"][:10]:
        lines.append(
            f"- idx `{row['sample_index']}` | score `{row['score']:.4f}` | "
            f"base `{row['exp_only_answer']}` | prob `{row['prob_vote']}` | "
            f"correct `{row['is_correct']}`"
        )

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote: {json_path}")
    print(f"Wrote: {txt_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
