"""Phase E6: concrete threshold-based fallback policy.

Turns the Phase E4 retry result into an operational policy:

    if reliability_score < tau:
        use prob_vote
    else:
        keep exp_only

Thresholds are selected on GSM8K validation data and then transferred
unchanged to GSM8K test and SVAMP.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import numpy as np

import analyze_reliability_retry as e4


DEFAULT_OUT_DIR = e4.DEFAULT_OUT_DIR
QUANTILES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def _apply_threshold_policy(rows, score_key, tau, fallback_key="prob_vote"):
    out = []
    n_flagged = 0
    n_changed = 0
    fixes = 0
    hurts = 0
    for r in rows:
        base = r["exp_only_answer"]
        cand = r.get(fallback_key)
        flagged = r[score_key] <= tau
        if flagged:
            n_flagged += 1
        use = cand if flagged and cand is not None else base
        if use != base:
            n_changed += 1
            if e4.e1._is_correct(use, r["ground_truth"]) and not e4.e1._is_correct(base, r["ground_truth"]):
                fixes += 1
            if not e4.e1._is_correct(use, r["ground_truth"]) and e4.e1._is_correct(base, r["ground_truth"]):
                hurts += 1
        out.append(use)
    acc = sum(e4.e1._is_correct(a, r["ground_truth"]) for a, r in zip(out, rows)) / len(rows)
    return {
        "tau": float(tau),
        "n": len(rows),
        "retry_rate": n_flagged / len(rows),
        "n_flagged": n_flagged,
        "n_changed": n_changed,
        "acc": acc,
        "delta_vs_base": acc - e4._accuracy(rows, "exp_only_answer"),
        "fixes": fixes,
        "hurts": hurts,
    }


def _candidate_thresholds(val_rows, score_key):
    scores = np.array([r[score_key] for r in val_rows], dtype=float)
    taus = []
    for q in QUANTILES:
        taus.append((q, float(np.quantile(scores, q))))
    return taus


def _select_best(val_rows, score_key):
    trials = []
    for q, tau in _candidate_thresholds(val_rows, score_key):
        res = _apply_threshold_policy(val_rows, score_key, tau)
        res["quantile"] = q
        trials.append(res)
    # maximize val acc, then prefer smaller retry rate
    best = max(trials, key=lambda r: (r["acc"], -r["retry_rate"]))
    return {"trials": trials, "best": best}


def _render_table(lines, title, trials):
    lines.append(f"## {title}")
    lines.append("")
    lines.append("| Quantile | tau | Retry rate | Changed | Acc | Delta | Fixes | Hurts |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in trials:
        lines.append(
            f"| {int(r['quantile']*100)}% | {r['tau']:.4f} | {r['retry_rate']:.2%} | {r['n_changed']} | {r['acc']:.2%} | {r['delta_vs_base']:+.2%} | {r['fixes']} | {r['hurts']} |"
        )
    lines.append("")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    gsm_feat = e4._load_feature_rows(e4.DEFAULT_GSM8K_BASE)
    gsm_base_ans = e4._load_answer_artifact(e4.DEFAULT_GSM8K_BASE)
    gsm_prob_ans = e4._load_answer_artifact(e4.DEFAULT_GSM8K_PROB)
    gsm_block_ans = e4._load_answer_artifact(e4.DEFAULT_GSM8K_BLOCKACTIVE)
    gsm_rows = e4._merge_sources(gsm_feat, gsm_base_ans, gsm_prob_ans, gsm_block_ans)

    svamp_feat = e4._load_feature_rows(e4.DEFAULT_SVAMP_BASE)
    svamp_base_ans = e4._load_answer_artifact(e4.DEFAULT_SVAMP_BASE)
    svamp_prob_ans = e4._load_answer_artifact(e4.DEFAULT_SVAMP_PROB)
    svamp_block_ans = e4._load_answer_artifact(e4.DEFAULT_SVAMP_BLOCKACTIVE)
    svamp_rows = e4._merge_sources(svamp_feat, svamp_base_ans, svamp_prob_ans, svamp_block_ans)

    split = e4._fit_scores(gsm_rows)
    gsm_val = split["val"]
    gsm_test = split["test"]
    # Recreate val-fitted scores on SVAMP, matching Phase E4/E5.
    zsum_spec = e4.e1._fit_combined_score(gsm_val, e4.e1.FEATURES)
    e4.e1._apply_combined_score(svamp_rows, zsum_spec)
    broad_features = split["zsum_features"]
    inner_train, inner_dev = e4.e2._inner_split(gsm_val)
    x_train, y_train, stats = e4.e2._prepare_matrix(inner_train, broad_features)
    best_l2 = None
    best_aurc = None
    for l2 in e4.e2.L2_GRID:
        model = e4.e2._fit_logreg_numpy(x_train, y_train, l2=l2)
        dev_scored = e4.e2._score_rows(
            inner_dev,
            "model_score",
            e4.e2._predict_scores(inner_dev, broad_features, stats, model),
        )
        aurc = e4.e2._curve_pack(dev_scored, "model_score")["aurc"]
        if best_aurc is None or aurc < best_aurc:
            best_aurc = aurc
            best_l2 = l2
    x_full, y_full, full_stats = e4.e2._prepare_matrix(gsm_val, broad_features)
    final_model = e4.e2._fit_logreg_numpy(x_full, y_full, l2=best_l2)
    svamp_model_scores = e4.e2._predict_scores(svamp_rows, broad_features, full_stats, final_model)
    for r, s in zip(svamp_rows, svamp_model_scores):
        r["logistic_broad_score"] = float(s)

    results = {"config": {"quantiles": QUANTILES}, "base_acc": {
        "gsm8k_val": e4._accuracy(gsm_val, "exp_only_answer"),
        "gsm8k_test": e4._accuracy(gsm_test, "exp_only_answer"),
        "svamp_full": e4._accuracy(svamp_rows, "exp_only_answer"),
        "prob_vote_gsm8k_test": e4._accuracy(gsm_test, "prob_vote"),
        "prob_vote_svamp": e4._accuracy(svamp_rows, "prob_vote"),
    }}

    for score_key in ["combined_score", "logistic_broad_score"]:
        sel = _select_best(gsm_val, score_key)
        best = sel["best"]
        test_eval = _apply_threshold_policy(gsm_test, score_key, best["tau"])
        svamp_eval = _apply_threshold_policy(svamp_rows, score_key, best["tau"])
        results[score_key] = {"val": sel, "test": test_eval, "svamp": svamp_eval}

    os.makedirs(args.out_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    json_path = os.path.join(args.out_dir, f"reliability_retry_policy_{date_str}.json")
    md_path = os.path.join(args.out_dir, f"reliability_retry_policy_{date_str}.md")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    lines = []
    lines.append(f"# Reliability Retry Policy (Phase E6) — {date_str}")
    lines.append("")
    lines.append("Operationalized fallback policy: if the reliability score is below a validation-selected threshold `tau`, replace `exp_only` with `prob_vote`.")
    lines.append("")
    lines.append("## Base reference")
    lines.append("")
    lines.append(f"- GSM8K val base: `{results['base_acc']['gsm8k_val']:.2%}`")
    lines.append(f"- GSM8K test base: `{results['base_acc']['gsm8k_test']:.2%}`")
    lines.append(f"- SVAMP base: `{results['base_acc']['svamp_full']:.2%}`")
    lines.append(f"- GSM8K test global `prob_vote`: `{results['base_acc']['prob_vote_gsm8k_test']:.2%}`")
    lines.append(f"- SVAMP global `prob_vote`: `{results['base_acc']['prob_vote_svamp']:.2%}`")
    lines.append("")
    for score_key in ["combined_score", "logistic_broad_score"]:
        _render_table(lines, f"{score_key} — GSM8K validation threshold search", results[score_key]["val"]["trials"])
        best = results[score_key]["val"]["best"]
        test = results[score_key]["test"]
        sv = results[score_key]["svamp"]
        lines.append(f"### {score_key} — chosen threshold")
        lines.append("")
        lines.append(f"- quantile: `{int(best['quantile']*100)}%`")
        lines.append(f"- tau: `{best['tau']:.4f}`")
        lines.append(f"- val retry rate: `{best['retry_rate']:.2%}`")
        lines.append(f"- GSM8K test: `{test['acc']:.2%}` (`{test['delta_vs_base']:+.2%}`), changed `{test['n_changed']}`, fixes `{test['fixes']}`, hurts `{test['hurts']}`")
        lines.append(f"- SVAMP: `{sv['acc']:.2%}` (`{sv['delta_vs_base']:+.2%}`), changed `{sv['n_changed']}`, fixes `{sv['fixes']}`, hurts `{sv['hurts']}`")
        lines.append("")
    lines.append("## Current read")
    lines.append("")
    lines.append("- This pass turns the retry budget result into a deployable score threshold.")
    lines.append("- If the thresholded policy matches the earlier budget-based result, that suggests the reliability score is stable enough to drive a simple fallback gate.")
    lines.append("- It is still an offline substitution test; a true retry policy would need fresh generation latency/cost accounting.")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
