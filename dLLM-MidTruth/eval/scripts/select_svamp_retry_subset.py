"""SVAMP retry-subset selection using the GSM8K-val-fitted reliability score.

Cross-task test for Phase E-Retry-CrossTask: keep the same score
definition and threshold as the GSM8K selective retry (logistic_broad
fit on GSM8K val, tau=0.5845), apply it to SVAMP samples, and select the
samples falling below tau. No SVAMP-specific tuning of the score or the
threshold.

Outputs (similar shape to the GSM8K selection script):
  - JSON manifest with metadata + per-sample rows
  - plain-text sample_indices.txt for the retry pipeline
  - short MD summary with ready GPU commands and the evaluation command
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
DEFAULT_TAU = 0.5845
SELECT_MODE_DEFAULT = "below_tau"


def _fit_score_on_gsm_val():
    gsm_feat = e4._load_feature_rows(e4.DEFAULT_GSM8K_BASE)
    gsm_base_ans = e4._load_answer_artifact(e4.DEFAULT_GSM8K_BASE)
    gsm_prob_ans = e4._load_answer_artifact(e4.DEFAULT_GSM8K_PROB)
    gsm_block_ans = e4._load_answer_artifact(e4.DEFAULT_GSM8K_BLOCKACTIVE)
    gsm_merged = e4._merge_sources(gsm_feat, gsm_base_ans, gsm_prob_ans, gsm_block_ans)
    gsm_val, _ = e1._split(gsm_merged)
    zsum_spec = e1._fit_combined_score(gsm_val, e1.FEATURES)
    broad_features = list(zsum_spec.keys())
    inner_train, inner_dev = e2._inner_split(gsm_val)
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
    x_full, y_full, full_stats = e2._prepare_matrix(gsm_val, broad_features)
    final_model = e2._fit_logreg_numpy(x_full, y_full, l2=best["l2"])
    return {
        "zsum_spec": zsum_spec,
        "broad_features": broad_features,
        "full_stats": full_stats,
        "final_model": final_model,
        "best_l2": best["l2"],
    }


def _load_svamp_scored(score):
    svamp_feat = e4._load_feature_rows(e4.DEFAULT_SVAMP_BASE)
    svamp_base_ans = e4._load_answer_artifact(e4.DEFAULT_SVAMP_BASE)
    svamp_prob_ans = e4._load_answer_artifact(e4.DEFAULT_SVAMP_PROB)
    svamp_block_ans = e4._load_answer_artifact(e4.DEFAULT_SVAMP_BLOCKACTIVE)
    merged = e4._merge_sources(svamp_feat, svamp_base_ans, svamp_prob_ans, svamp_block_ans)
    e1._apply_combined_score(merged, score["zsum_spec"])
    scores = e2._predict_scores(merged, score["broad_features"], score["full_stats"], score["final_model"])
    for r, s in zip(merged, scores):
        r["logistic_broad_score"] = float(s)
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-key", choices=["combined_score", "logistic_broad_score"], default=DEFAULT_SCORE_KEY)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU,
                        help="threshold from GSM8K val; samples with score <= tau are flagged")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--output-stem", default="")
    args = parser.parse_args()

    print(f"[fit] reliability score on GSM8K val (same as GSM8K selective retry)")
    score = _fit_score_on_gsm_val()
    print(f"  best L2 = {score['best_l2']}")

    print(f"[load] SVAMP merged + apply score")
    svamp = _load_svamp_scored(score)
    print(f"  n = {len(svamp)}")

    flagged = [r for r in svamp if float(r[args.score_key]) <= args.tau]
    base_correct = sum(r["is_correct"] for r in svamp)
    flagged_correct = sum(r["is_correct"] for r in flagged)
    overall_base = base_correct / len(svamp)
    flagged_base = flagged_correct / len(flagged) if flagged else 0
    print(f"  tau = {args.tau}")
    print(f"  flagged: {len(flagged)}/{len(svamp)} = {len(flagged)/len(svamp)*100:.2f}%  base_acc={flagged_base*100:.2f}%")
    print(f"  overall base_acc = {overall_base*100:.2f}%")

    date_str = datetime.now().strftime("%Y%m%d")
    stem = args.output_stem or f"svamp_retry_subset_{args.score_key}_tau{int(args.tau*10000):05d}_{date_str}"
    os.makedirs(args.out_dir, exist_ok=True)

    manifest = {
        "base_artifact": e4.DEFAULT_SVAMP_BASE,
        "score_key": args.score_key,
        "tau": args.tau,
        "score_fit_source": "GSM8K val (seed=42, frac=0.8)",
        "subset_split": "svamp_full",
        "n_total": len(svamp),
        "n_selected": len(flagged),
        "retry_rate": len(flagged) / len(svamp) if svamp else 0.0,
        "overall_base_acc": overall_base,
        "flagged_base_acc": flagged_base,
        "unflagged_n": len(svamp) - len(flagged),
        "unflagged_base_acc": (base_correct - flagged_correct) / (len(svamp) - len(flagged)) if (len(svamp) - len(flagged)) else 0,
        "sample_indices": [int(r["sample_index"]) for r in flagged],
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
            for r in flagged
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
    lines.append(f"# SVAMP Selective Retry Subset — {date_str}")
    lines.append("")
    lines.append("Cross-task transfer of the GSM8K-val-fitted reliability score: same `tau`, same score definition, applied to SVAMP samples. No SVAMP-specific tuning.")
    lines.append("")
    lines.append(f"- base artifact: `{e4.DEFAULT_SVAMP_BASE}`")
    lines.append(f"- score fit source: GSM8K val (seed=42, frac=0.8)")
    lines.append(f"- score key: `{args.score_key}`")
    lines.append(f"- tau: `{args.tau:.4f}` (GSM8K val 25%-quantile of logistic_broad_score)")
    lines.append(f"- SVAMP overall base acc: `{overall_base*100:.2f}%`")
    lines.append(f"- selected (flagged): `{manifest['n_selected']}/{manifest['n_total']}` (`{manifest['retry_rate']*100:.2f}%`)")
    lines.append(f"  - flagged base acc: `{flagged_base*100:.2f}%`")
    lines.append(f"  - unflagged base acc: `{manifest['unflagged_base_acc']*100:.2f}%`")
    lines.append(f"- json: `{json_path}`")
    lines.append(f"- txt:  `{txt_path}`")
    lines.append("")
    lines.append("## Use — GPU retry on flagged subset")
    lines.append("")
    lines.append("```")
    lines.append("cd /home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval")
    lines.append("")
    lines.append("PYTHON_BIN=/home/work/GFlowPO/anaconda3/envs/prophet/bin/python \\")
    lines.append(f"SUBSET_INDICES_FILE={txt_path} \\")
    lines.append(f"RUN_NAME=20260515_svamp_retry_exp_selective_tau{int(args.tau*10000):05d} \\")
    lines.append("TASK=svamp \\")
    lines.append("VOTE_METHOD=exp \\")
    lines.append("bash scripts/run_retry_policy_experiment.sh 1")
    lines.append("```")
    lines.append("")
    lines.append("Trailing `1` is the GPU index. Expected wall-time `~25s` based on the 60-sample GSM8K retry baseline (SVAMP is shorter prompts).")
    lines.append("")
    lines.append("## Next: complement-only random control")
    lines.append("")
    lines.append("Generate a budget-matched random control from the unflagged SVAMP slice:")
    lines.append("")
    lines.append("```")
    lines.append(f"/home/work/GFlowPO/anaconda3/envs/prophet/bin/python scripts/select_svamp_random_retry_subset.py --random-seed 124 --exclude-flagged --tau {args.tau} --k {manifest['n_selected']}")
    lines.append("```")
    lines.append("")
    lines.append("Then run the random GPU retry on that manifest, evaluate both with `evaluate_retry_control.py` adapted for SVAMP, or use a SVAMP-aware evaluator (see follow-up script).")
    lines.append("")
    lines.append("## First 10 selected SVAMP samples")
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
