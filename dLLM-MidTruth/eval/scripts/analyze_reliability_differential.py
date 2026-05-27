"""Differential targetability check for the reliability score (followup).

Motivation. The E4 retry simulation shows that reliability-gated retry with
`prob_vote` as the source yields GSM8K test `67.42% -> 68.94%` (`+1.52pt`).
But applying `prob_vote` to *every* GSM8K test sample yields `70.08%`
(`+2.66pt`). So on GSM8K test the gating is `+1.14pt` worse than the global
swap. This script asks the underlying question directly:

  Does the reliability score predict samples where `prob_vote` is right
  but `exp_only` is wrong (`delta = +1`), as opposed to merely capturing
  generic sample difficulty?

If yes, the gating should beat random retry of the same budget. If no,
gating just trades sample selection cost for no targeting benefit, and
"use prob_vote globally" becomes the right operational baseline.

Analyses
- delta = is_correct(prob_vote) - is_correct(exp_only) ∈ {-1, 0, +1}
- Pearson(score, delta), Pearson(score, y_exp), Pearson(score, y_prob)
- Score-quantile breakdown: fraction of {prob_better, tie, exp_better}
- Net swap value (fixes - hurts) at each retry budget, vs random retry
  and vs oracle retry (sort by delta).
- Operational strategy comparison: exp_only / prob_vote globally /
  reliability-gated / random-gated / oracle-gated.

Outputs
  eval/analysis/reliability_differential_{date}.json
  eval/analysis/reliability_differential_{date}.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import scripts.analyze_reliability as e1  # noqa: E402
import scripts.analyze_reliability_retry as retry  # noqa: E402


DEFAULT_OUT_DIR = retry.DEFAULT_OUT_DIR
SEED = 42

BUDGET_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def _delta(row):
    y_exp = 1 if e1._is_correct(row["exp_only_answer"], row["ground_truth"]) else 0
    pv = row.get("prob_vote")
    if pv is None:
        # treat missing prob_vote as no swap signal: tie at exp value
        return 0, y_exp, y_exp
    y_prob = 1 if e1._is_correct(pv, row["ground_truth"]) else 0
    return y_prob - y_exp, y_exp, y_prob


def _enrich(rows):
    out = []
    for r in rows:
        d, ye, yp = _delta(r)
        out.append({**r, "delta": d, "y_exp": ye, "y_prob": yp})
    return out


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    vy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if vx == 0 or vy == 0:
        return 0.0
    return cov / (vx * vy)


def _score_corrs(rows, score_key):
    xs = [r[score_key] for r in rows]
    return {
        "score_vs_delta": _pearson(xs, [r["delta"] for r in rows]),
        "score_vs_y_exp": _pearson(xs, [r["y_exp"] for r in rows]),
        "score_vs_y_prob": _pearson(xs, [r["y_prob"] for r in rows]),
    }


def _quantile_breakdown(rows, score_key, nq=5):
    if not rows:
        return []
    ordered = sorted(rows, key=lambda r: r[score_key])
    n = len(ordered)
    bucket_size = n / nq
    out = []
    for q in range(nq):
        lo = int(round(q * bucket_size))
        hi = int(round((q + 1) * bucket_size))
        chunk = ordered[lo:hi]
        if not chunk:
            continue
        pb = sum(1 for r in chunk if r["delta"] == 1)
        eb = sum(1 for r in chunk if r["delta"] == -1)
        tie = len(chunk) - pb - eb
        out.append({
            "quantile_idx": q,
            "score_range": [chunk[0][score_key], chunk[-1][score_key]],
            "n": len(chunk),
            "prob_better": pb,
            "tie": tie,
            "exp_better": eb,
            "net_swap_value": pb - eb,
            "y_exp_acc": sum(r["y_exp"] for r in chunk) / len(chunk),
            "y_prob_acc": sum(r["y_prob"] for r in chunk) / len(chunk),
        })
    return out


def _swap_strategies(rows, score_key):
    """Compute net swap (fixes - hurts) at each budget under:
    - score-gated (sort by score ascending, swap bottom k)
    - random-gated (random sort, swap bottom k)
    - oracle-gated (sort by -delta then delta, swap bottom k)
    Returns a dict keyed by budget.
    """
    n = len(rows)
    if n == 0:
        return {}

    def _net_with_order(order, k):
        targets = order[:k]
        fixes = sum(1 for r in targets if r["delta"] == 1)
        hurts = sum(1 for r in targets if r["delta"] == -1)
        return fixes, hurts, fixes - hurts

    score_order = sorted(rows, key=lambda r: r[score_key])
    rng = random.Random(SEED)
    rand_order = list(rows)
    rng.shuffle(rand_order)
    # Oracle: sort by delta descending so prob_better (+1) come first,
    # exp_better (-1) come last. Tie samples in the middle, neutral.
    oracle_order = sorted(rows, key=lambda r: -r["delta"])

    base_acc = sum(r["y_exp"] for r in rows) / n
    out = {}
    for frac in BUDGET_GRID:
        k = max(1, int(round(frac * n)))
        s_fix, s_hurt, s_net = _net_with_order(score_order, k)
        r_fix, r_hurt, r_net = _net_with_order(rand_order, k)
        o_fix, o_hurt, o_net = _net_with_order(oracle_order, k)
        out[frac] = {
            "k": k,
            "score_gated":  {"fixes": s_fix, "hurts": s_hurt, "net": s_net,
                              "acc": base_acc + s_net / n},
            "random_gated": {"fixes": r_fix, "hurts": r_hurt, "net": r_net,
                              "acc": base_acc + r_net / n},
            "oracle_gated": {"fixes": o_fix, "hurts": o_hurt, "net": o_net,
                              "acc": base_acc + o_net / n},
        }
    return out


def _overall_strategy_accs(rows):
    n = len(rows)
    exp_acc = sum(r["y_exp"] for r in rows) / n
    prob_acc = sum(r["y_prob"] for r in rows) / n
    # union ceiling: a sample contributes if exp or prob (or both) is right
    union_acc = sum(1 for r in rows if r["y_exp"] or r["y_prob"]) / n
    intersection = sum(1 for r in rows if r["y_exp"] and r["y_prob"]) / n
    pb = sum(1 for r in rows if r["delta"] == 1)
    eb = sum(1 for r in rows if r["delta"] == -1)
    return {
        "n": n,
        "exp_only_acc": exp_acc,
        "prob_vote_acc": prob_acc,
        "union_oracle_acc": union_acc,
        "intersection_both_correct": intersection,
        "prob_better_count": pb,
        "exp_better_count": eb,
        "delta_zero_count": n - pb - eb,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gsm8k-base", default=retry.DEFAULT_GSM8K_BASE)
    parser.add_argument("--svamp-base", default=retry.DEFAULT_SVAMP_BASE)
    parser.add_argument("--gsm8k-prob", default=retry.DEFAULT_GSM8K_PROB)
    parser.add_argument("--svamp-prob", default=retry.DEFAULT_SVAMP_PROB)
    parser.add_argument("--gsm8k-block", default=retry.DEFAULT_GSM8K_BLOCKACTIVE)
    parser.add_argument("--svamp-block", default=retry.DEFAULT_SVAMP_BLOCKACTIVE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    print(f"[load] GSM8K base: {args.gsm8k_base}")
    gsm_feature_rows = retry._load_feature_rows(args.gsm8k_base)
    gsm_base_ans = retry._load_answer_artifact(args.gsm8k_base)
    gsm_prob_ans = retry._load_answer_artifact(args.gsm8k_prob)
    gsm_block_ans = retry._load_answer_artifact(args.gsm8k_block)
    gsm_merged = retry._merge_sources(gsm_feature_rows, gsm_base_ans, gsm_prob_ans, gsm_block_ans)

    print(f"[load] SVAMP base: {args.svamp_base}")
    svamp_feature_rows = retry._load_feature_rows(args.svamp_base)
    svamp_base_ans = retry._load_answer_artifact(args.svamp_base)
    svamp_prob_ans = retry._load_answer_artifact(args.svamp_prob)
    svamp_block_ans = retry._load_answer_artifact(args.svamp_block)
    svamp_merged = retry._merge_sources(svamp_feature_rows, svamp_base_ans, svamp_prob_ans, svamp_block_ans)

    print(f"[score] fitting scores on GSM8K val (seed={SEED})...")
    fit = retry._fit_scores(gsm_merged)
    gsm_val = fit["val"]
    gsm_test = fit["test"]

    # Rebuild SVAMP scores using val-fitted specs (mirrors retry script logic).
    zsum_spec = e1._fit_combined_score(gsm_val, e1.FEATURES)
    e1._apply_combined_score(svamp_merged, zsum_spec)
    # logistic_broad on SVAMP
    import scripts.analyze_reliability_v2 as e2  # noqa: E402
    broad_features = list(zsum_spec.keys())
    x_full, _, full_stats = e2._prepare_matrix(gsm_val, broad_features)
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
    svamp_scores = e2._predict_scores(svamp_merged, broad_features, full_stats, final_model)
    for r, s in zip(svamp_merged, svamp_scores):
        r["logistic_broad_score"] = float(s)

    # Enrich with delta info
    gsm_val = _enrich(gsm_val)
    gsm_test = _enrich(gsm_test)
    svamp_enriched = _enrich(svamp_merged)

    score_keys = ["combined_score", "logistic_broad_score"]

    splits = {
        "gsm8k_val": gsm_val,
        "gsm8k_test": gsm_test,
        "svamp": svamp_enriched,
    }

    overall = {name: _overall_strategy_accs(rows) for name, rows in splits.items()}

    corrs = {}
    quant = {}
    swaps = {}
    for name, rows in splits.items():
        corrs[name] = {sk: _score_corrs(rows, sk) for sk in score_keys}
        quant[name] = {sk: _quantile_breakdown(rows, sk, nq=5) for sk in score_keys}
        swaps[name] = {sk: _swap_strategies(rows, sk) for sk in score_keys}

    # === Output ===
    os.makedirs(args.out_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    stem = f"reliability_differential_{date_str}"

    out = {
        "config": {"seed": SEED, "budget_grid": BUDGET_GRID, "score_keys": score_keys},
        "sources": {
            "gsm8k_base": args.gsm8k_base, "gsm8k_prob": args.gsm8k_prob,
            "svamp_base": args.svamp_base, "svamp_prob": args.svamp_prob,
        },
        "overall": overall,
        "score_corrs": corrs,
        "quantile_breakdown": quant,
        "swap_strategies": swaps,
    }
    json_path = os.path.join(args.out_dir, f"{stem}.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote: {json_path}")

    # Markdown summary
    lines = []
    lines.append(f"# Reliability Score — Differential Targetability Followup — {date_str}")
    lines.append("")
    lines.append("Asks whether the reliability score predicts samples where `prob_vote` beats `exp_only` (delta = +1), as opposed to capturing generic difficulty. Pairs with `reliability_retry_20260514.md` and `reliability_retry_policy_20260514.md`.")
    lines.append("")
    lines.append("## Strategy accuracies (global, no gating)")
    lines.append("")
    lines.append("| Split | n | exp_only | prob_vote | union oracle | both correct | prob_better | exp_better |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name in ["gsm8k_val", "gsm8k_test", "svamp"]:
        o = overall[name]
        lines.append(f"| {name} | {o['n']} | {o['exp_only_acc']*100:.2f}% | {o['prob_vote_acc']*100:.2f}% | {o['union_oracle_acc']*100:.2f}% | {o['intersection_both_correct']*100:.2f}% | {o['prob_better_count']} | {o['exp_better_count']} |")
    lines.append("")
    lines.append("**Read**: on a split where `prob_vote` global > `exp_only` global, the right baseline before any reliability gating is the global swap. The reliability score's job is to beat that, not just to beat `exp_only`.")
    lines.append("")

    for score_key in score_keys:
        lines.append(f"## Score: `{score_key}`")
        lines.append("")
        lines.append("### Score vs (delta / y_exp / y_prob) Pearson")
        lines.append("")
        lines.append("| Split | Pearson(score, delta) | Pearson(score, y_exp) | Pearson(score, y_prob) |")
        lines.append("|---|---:|---:|---:|")
        for name in ["gsm8k_val", "gsm8k_test", "svamp"]:
            c = corrs[name][score_key]
            lines.append(f"| {name} | {c['score_vs_delta']:+.3f} | {c['score_vs_y_exp']:+.3f} | {c['score_vs_y_prob']:+.3f} |")
        lines.append("")
        lines.append("**Read**: large |Pearson(score, y_exp)| with small |Pearson(score, delta)| means the score captures generic difficulty, not method-switch signal.")
        lines.append("")

        lines.append("### Quantile breakdown (5 buckets, low score → high score)")
        lines.append("")
        for name in ["gsm8k_val", "gsm8k_test", "svamp"]:
            lines.append(f"#### {name}")
            lines.append("")
            lines.append("| Q | n | exp_acc | prob_acc | prob_better | tie | exp_better | net swap value |")
            lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
            for q in quant[name][score_key]:
                lines.append(f"| {q['quantile_idx']} | {q['n']} | {q['y_exp_acc']*100:.2f}% | {q['y_prob_acc']*100:.2f}% | {q['prob_better']} | {q['tie']} | {q['exp_better']} | {q['net_swap_value']:+d} |")
            lines.append("")

        lines.append("### Net swap (fixes - hurts) at each budget")
        lines.append("")
        for name in ["gsm8k_val", "gsm8k_test", "svamp"]:
            lines.append(f"#### {name}")
            lines.append("")
            lines.append("| budget | k | score-gated net (acc) | random-gated net (acc) | oracle-gated net (acc) |")
            lines.append("|---:|---:|---:|---:|---:|")
            for frac in BUDGET_GRID:
                s = swaps[name][score_key][frac]
                sg = s["score_gated"]; rg = s["random_gated"]; og = s["oracle_gated"]
                lines.append(
                    f"| {frac*100:.0f}% | {s['k']} | "
                    f"{sg['net']:+d} ({sg['acc']*100:.2f}%) | "
                    f"{rg['net']:+d} ({rg['acc']*100:.2f}%) | "
                    f"{og['net']:+d} ({og['acc']*100:.2f}%) |"
                )
            lines.append("")
        lines.append("**Read**: score-gated > random-gated means the reliability ranking adds targeting value over a random budget at the same size. score-gated < random-gated means the gating is actively misleading. The oracle column shows the ceiling for that budget.")
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- Net swap at the operational E6 budget (`~25%` for `logistic_broad`) on `gsm8k_test` is the most directly comparable point to the existing E4 / E6 result.")
    lines.append("- A clean negative (score-gated ≤ random-gated AND prob_vote global ≥ score-gated swap) would mean E4/E6's published lift comes from prob_vote being better on average rather than from any targeting effect.")
    lines.append("- A clean positive (score-gated > random-gated AND ≥ prob_vote global) would justify keeping the operational threshold rule as the recommended policy.")
    lines.append("- This pass is offline. No new generation. Same artifacts as Phase E/E2/E4/E5/E6.")
    md_path = os.path.join(args.out_dir, f"{stem}.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
