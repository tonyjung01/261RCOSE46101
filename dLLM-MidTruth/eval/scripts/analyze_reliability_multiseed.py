"""Phase E-Diff2: multi-seed outer-split robustness pass.

The E6 threshold rule and the E-Diff comparison both used a single outer
split (seed=42). gsm8k_test under that seed has `exp_better=0`, which is
the load-bearing distributional artifact behind the reported `+1.52pt`.
This script loops N outer seeds, refits the same logistic_broad pipeline
on val per seed, picks tau from val, and reports per-seed:

- exp_only test accuracy
- prob_vote test accuracy (global swap)
- threshold-gated test accuracy (E6 rule with seed-local tau)
- prob_better / exp_better counts on test

Then it aggregates mean ± std and reports the fraction of seeds in which:

- gated beats exp_only
- prob_vote globally beats gated
- gated beats prob_vote globally

Outputs
  eval/analysis/reliability_multiseed_{date}.{md,json}
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
import scripts.analyze_reliability_v2 as e2  # noqa: E402
import scripts.analyze_reliability_retry as retry  # noqa: E402


DEFAULT_OUT_DIR = retry.DEFAULT_OUT_DIR
DEFAULT_SEEDS = list(range(42, 52))  # 10 seeds
THRESHOLD_QUANTILE = 0.25  # E6 chose 25% retry budget


def _split_with_seed(rows, frac, seed):
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    n_val = int(len(rows) * frac)
    val_ids = set(indices[:n_val])
    val = [r for r in rows if r["sample_index"] in val_ids]
    test = [r for r in rows if r["sample_index"] not in val_ids]
    return val, test


def _fit_logistic_broad(val_rows, broad_features):
    inner_train, inner_dev = e2._inner_split(val_rows)
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
    x_full, y_full, full_stats = e2._prepare_matrix(val_rows, broad_features)
    final_model = e2._fit_logreg_numpy(x_full, y_full, l2=best["l2"])
    return final_model, full_stats, broad_features, best["l2"]


def _score(rows, model, stats, broad_features):
    s = e2._predict_scores(rows, broad_features, stats, model)
    out = []
    for r, sc in zip(rows, s):
        out.append({**r, "logistic_broad_score": float(sc)})
    return out


def _gated_test_acc(test_rows, tau):
    correct = 0
    fixes = 0
    hurts = 0
    changed = 0
    for r in test_rows:
        base = r["exp_only_answer"]
        pv = r.get("prob_vote")
        use = pv if (r["logistic_broad_score"] < tau and pv is not None) else base
        if use != base:
            changed += 1
            if e1._is_correct(use, r["ground_truth"]) and not e1._is_correct(base, r["ground_truth"]):
                fixes += 1
            if not e1._is_correct(use, r["ground_truth"]) and e1._is_correct(base, r["ground_truth"]):
                hurts += 1
        if e1._is_correct(use, r["ground_truth"]):
            correct += 1
    return correct / len(test_rows), {"changed": changed, "fixes": fixes, "hurts": hurts}


def _acc(rows, key):
    return sum(1 for r in rows if e1._is_correct(r.get(key), r["ground_truth"])) / len(rows)


def _delta_counts(rows):
    pb = sum(
        1 for r in rows
        if e1._is_correct(r.get("prob_vote"), r["ground_truth"])
        and not e1._is_correct(r["exp_only_answer"], r["ground_truth"])
    )
    eb = sum(
        1 for r in rows
        if not e1._is_correct(r.get("prob_vote"), r["ground_truth"])
        and e1._is_correct(r["exp_only_answer"], r["ground_truth"])
    )
    return pb, eb


def _mean_std(xs):
    n = len(xs)
    if n == 0:
        return None, None
    m = sum(xs) / n
    if n == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gsm8k-base", default=retry.DEFAULT_GSM8K_BASE)
    parser.add_argument("--gsm8k-prob", default=retry.DEFAULT_GSM8K_PROB)
    parser.add_argument("--gsm8k-block", default=retry.DEFAULT_GSM8K_BLOCKACTIVE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    args = parser.parse_args()

    print(f"[load] GSM8K base / prob / block (single load, multi-seed split)")
    gsm_feature_rows = retry._load_feature_rows(args.gsm8k_base)
    gsm_base_ans = retry._load_answer_artifact(args.gsm8k_base)
    gsm_prob_ans = retry._load_answer_artifact(args.gsm8k_prob)
    gsm_block_ans = retry._load_answer_artifact(args.gsm8k_block)
    merged = retry._merge_sources(gsm_feature_rows, gsm_base_ans, gsm_prob_ans, gsm_block_ans)
    print(f"  n={len(merged)}")

    per_seed = []
    for seed in args.seeds:
        val, test = _split_with_seed(merged, e1.VALIDATION_FRAC, seed)
        zsum_spec = e1._fit_combined_score(val, e1.FEATURES)
        broad_features = list(zsum_spec.keys())
        model, stats, _, l2 = _fit_logistic_broad(val, broad_features)
        val = _score(val, model, stats, broad_features)
        test = _score(test, model, stats, broad_features)

        # tau from val at the 25% retry quantile (matches E6)
        scores_val = sorted(r["logistic_broad_score"] for r in val)
        k = int(round(THRESHOLD_QUANTILE * len(scores_val)))
        # use the score at index k-1 (inclusive end of the bottom k) as the
        # threshold so exactly k val samples satisfy score < tau (approximately)
        tau = scores_val[max(0, k - 1)]

        exp_acc = _acc(test, "exp_only_answer")
        prob_acc = _acc(test, "prob_vote")
        gated_acc, gated_info = _gated_test_acc(test, tau)
        pb, eb = _delta_counts(test)

        per_seed.append({
            "seed": seed,
            "n_val": len(val),
            "n_test": len(test),
            "l2_chosen": l2,
            "tau": tau,
            "exp_only_acc": exp_acc,
            "prob_vote_acc": prob_acc,
            "gated_acc": gated_acc,
            "gated_changed": gated_info["changed"],
            "gated_fixes": gated_info["fixes"],
            "gated_hurts": gated_info["hurts"],
            "prob_better_count": pb,
            "exp_better_count": eb,
            "gated_minus_exp": gated_acc - exp_acc,
            "prob_minus_exp": prob_acc - exp_acc,
            "gated_minus_prob": gated_acc - prob_acc,
        })
        print(
            f"  seed={seed:>3}: exp={exp_acc*100:.2f}%  prob={prob_acc*100:.2f}%  "
            f"gated={gated_acc*100:.2f}%  pb/eb={pb}/{eb}  fixes/hurts={gated_info['fixes']}/{gated_info['hurts']}"
        )

    # === Aggregate ===
    exp_accs = [p["exp_only_acc"] for p in per_seed]
    prob_accs = [p["prob_vote_acc"] for p in per_seed]
    gated_accs = [p["gated_acc"] for p in per_seed]
    pb_counts = [p["prob_better_count"] for p in per_seed]
    eb_counts = [p["exp_better_count"] for p in per_seed]
    gate_minus_exp = [p["gated_minus_exp"] for p in per_seed]
    prob_minus_exp = [p["prob_minus_exp"] for p in per_seed]
    gate_minus_prob = [p["gated_minus_prob"] for p in per_seed]

    m_exp, sd_exp = _mean_std(exp_accs)
    m_prob, sd_prob = _mean_std(prob_accs)
    m_gated, sd_gated = _mean_std(gated_accs)
    m_pb, sd_pb = _mean_std(pb_counts)
    m_eb, sd_eb = _mean_std(eb_counts)

    n_seeds = len(per_seed)
    pct_gated_beats_exp = sum(1 for d in gate_minus_exp if d > 0) / n_seeds
    pct_prob_beats_exp = sum(1 for d in prob_minus_exp if d > 0) / n_seeds
    pct_prob_beats_gated = sum(1 for d in gate_minus_prob if d < 0) / n_seeds
    pct_gated_beats_prob = sum(1 for d in gate_minus_prob if d > 0) / n_seeds

    aggregate = {
        "n_seeds": n_seeds,
        "exp_only_acc_mean_std": [m_exp, sd_exp],
        "prob_vote_acc_mean_std": [m_prob, sd_prob],
        "gated_acc_mean_std": [m_gated, sd_gated],
        "gated_minus_exp_mean_std": [_mean_std(gate_minus_exp)[0], _mean_std(gate_minus_exp)[1]],
        "prob_minus_exp_mean_std": [_mean_std(prob_minus_exp)[0], _mean_std(prob_minus_exp)[1]],
        "gated_minus_prob_mean_std": [_mean_std(gate_minus_prob)[0], _mean_std(gate_minus_prob)[1]],
        "prob_better_mean_std": [m_pb, sd_pb],
        "exp_better_mean_std": [m_eb, sd_eb],
        "pct_seeds_gated_beats_exp": pct_gated_beats_exp,
        "pct_seeds_prob_beats_exp": pct_prob_beats_exp,
        "pct_seeds_prob_beats_gated": pct_prob_beats_gated,
        "pct_seeds_gated_beats_prob": pct_gated_beats_prob,
    }

    # === Output ===
    os.makedirs(args.out_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    stem = f"reliability_multiseed_{date_str}"

    out = {
        "config": {
            "seeds": args.seeds,
            "validation_frac": e1.VALIDATION_FRAC,
            "threshold_quantile": THRESHOLD_QUANTILE,
        },
        "sources": {"gsm8k_base": args.gsm8k_base, "gsm8k_prob": args.gsm8k_prob},
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    json_path = os.path.join(args.out_dir, f"{stem}.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote: {json_path}")

    lines = []
    lines.append(f"# E-Diff2 — Multi-seed Outer-Split Robustness — {date_str}")
    lines.append("")
    lines.append(f"Seeds: {args.seeds}; val_frac={e1.VALIDATION_FRAC}; threshold_quantile={THRESHOLD_QUANTILE}")
    lines.append("")
    lines.append("## Per-seed results on GSM8K")
    lines.append("")
    lines.append("| Seed | n_test | exp_only | prob_vote | gated | prob_better | exp_better | fixes | hurts |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for p in per_seed:
        lines.append(
            f"| {p['seed']} | {p['n_test']} | {p['exp_only_acc']*100:.2f}% | "
            f"{p['prob_vote_acc']*100:.2f}% | {p['gated_acc']*100:.2f}% | "
            f"{p['prob_better_count']} | {p['exp_better_count']} | "
            f"{p['gated_fixes']} | {p['gated_hurts']} |"
        )
    lines.append("")

    lines.append("## Aggregate")
    lines.append("")
    lines.append(f"- n_seeds = {n_seeds}")
    lines.append(f"- exp_only test acc:   `{m_exp*100:.2f}% ± {sd_exp*100:.2f}pt`")
    lines.append(f"- prob_vote test acc:  `{m_prob*100:.2f}% ± {sd_prob*100:.2f}pt`")
    lines.append(f"- gated test acc:      `{m_gated*100:.2f}% ± {sd_gated*100:.2f}pt`")
    lines.append("")
    lines.append("### Deltas")
    lines.append("")
    gme, gms = _mean_std(gate_minus_exp)
    pme, pms = _mean_std(prob_minus_exp)
    gmp, gpms = _mean_std(gate_minus_prob)
    lines.append(f"- gated − exp_only:  `{gme*100:+.2f}pt ± {gms*100:.2f}pt`")
    lines.append(f"- prob_vote − exp_only: `{pme*100:+.2f}pt ± {pms*100:.2f}pt`")
    lines.append(f"- gated − prob_vote: `{gmp*100:+.2f}pt ± {gpms*100:.2f}pt`")
    lines.append("")
    lines.append("### Seed-level outcomes")
    lines.append("")
    lines.append(f"- Fraction of seeds where gated beats exp_only: **{pct_gated_beats_exp*100:.0f}%**")
    lines.append(f"- Fraction of seeds where prob_vote beats exp_only: **{pct_prob_beats_exp*100:.0f}%**")
    lines.append(f"- Fraction of seeds where prob_vote beats gated: **{pct_prob_beats_gated*100:.0f}%**")
    lines.append(f"- Fraction of seeds where gated beats prob_vote: **{pct_gated_beats_prob*100:.0f}%**")
    lines.append("")
    lines.append(f"- prob_better mean ± std: `{m_pb:.2f} ± {sd_pb:.2f}`")
    lines.append(f"- exp_better mean ± std:  `{m_eb:.2f} ± {sd_eb:.2f}`")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- The seed=42 result reported in E4/E6 (`+1.52pt`) corresponds to a single row above; compare it against the mean to see how outlier-like it is.")
    lines.append("- If the `gated − prob_vote` mean is consistently negative across seeds, that confirms the differential targetability finding: the threshold rule is dominated by global `prob_vote` on average, and the seed=42 result was a lucky single split.")
    lines.append("- If `prob_better − exp_better` is approximately constant across seeds, the test split's `exp_better=0` at seed=42 is the outlier driver.")

    md_path = os.path.join(args.out_dir, f"{stem}.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
