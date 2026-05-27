"""Phase E (exploratory): sample-level reliability for selective abstention.

Motivation. Phase 8 offline proxy showed that per-sample features derived
from the gap/answer trajectory correlate with baseline (`exp_only`)
correctness much more strongly than the diffuse mean gap:

  max_gap                Pearson +0.332
  last_change_step_frac  Pearson -0.308
  n_unique_answers       Pearson -0.277

These are **between-sample** reliability signals, not within-sample vote
weights. This script asks whether they can drive a useful selective
abstention policy: rank samples by reliability, abstain on the bottom,
measure the lift in selective accuracy on the remainder.

Protocol (mirrors Phase 7 test-leakage rules)
- Sample split: seed=42, 80/20 (same as Phase 7).
- Single-feature ranking baselines: evaluated on val and test separately.
- Combined score: z-score normalization with means/stds fitted on val,
  feature signs taken from val-set Pearson sign; the score is the sum
  of signed z-scores. Test/SVAMP only see the val-fitted score.
- Reference curves: random (no signal), oracle (sort by ground-truth
  correctness — empirical ceiling).

Outputs
  eval/analysis/reliability_abstention_{date}.json
  eval/analysis/reliability_abstention_{date}.md
"""

import argparse
import json
import math
import os
import random
import sys
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

DEFAULT_GSM8K = os.path.join(
    REPO_ROOT,
    "outputs/LLaDA-8B-Instruct/20260429_everpass_debug_bs4/"
    "gsm8k_gen128_steps64_vote_confidence_gap_answer_window5_mean_rawsum_bs4_debug/"
    "rank_0_generations.json",
)
DEFAULT_SVAMP = os.path.join(
    REPO_ROOT,
    "outputs/LLaDA-8B-Instruct/20260429_everpass_debug_bs4/"
    "svamp_gen128_steps64_vote_confidence_gap_answer_window5_mean_rawsum_bs4_debug/"
    "rank_0_generations.json",
)
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "analysis")

SEED = 42
VALIDATION_FRAC = 0.8
ALPHA = 5.0

COVERAGES = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5]

# Single-feature candidates. The script will auto-orient each one by the
# val-set Pearson sign so that a higher score always means "more reliable".
FEATURES = [
    "max_gap",
    "mean_gap",
    "last_change_gap",
    "first_appear_gap",
    "std_gap",
    "n_valid",
    "n_unique_answers",
    "last_change_step_frac",
    "max_gap_step_frac",
    "first_appear_step_frac",
]


def _is_correct(vote, gt, atol=1e-6):
    if vote is None or gt is None:
        return False
    try:
        return abs(float(vote) - float(gt)) < atol
    except (TypeError, ValueError):
        return str(vote) == str(gt)


def _exp_only_vote(valid):
    if not valid:
        return None
    scores = {}
    for ev in valid:
        scores[ev["parsed_answer"]] = scores.get(ev["parsed_answer"], 0.0) + ev["exp_weight"]
    if not scores:
        return None
    return max(scores.items(), key=lambda kv: kv[1])[0]


def _load(path):
    with open(path) as f:
        data = json.load(f)
    rows = []
    for i, vd in enumerate(data["vote_debug"]):
        gt = data["generations"][i]["ground_truth"]
        valid = []
        for step in vd["steps"]:
            if step.get("skip_reason") is not None:
                continue
            rw = step.get("raw_weight")
            pa = step.get("parsed_answer")
            if rw is None or pa is None:
                continue
            valid.append({
                "step": step["step"],
                "total_steps": step["total_steps"],
                "gap": float(rw),
                "parsed_answer": pa,
                "exp_weight": math.exp(step["step"] / step["total_steps"] * ALPHA),
            })
        if not valid:
            # Samples with no valid event get a default "uninformative"
            # feature row so they don't silently disappear from the
            # coverage denominator.
            rows.append({
                "sample_index": i,
                "is_correct": 0,
                "max_gap": 0.0,
                "mean_gap": 0.0,
                "last_change_gap": 0.0,
                "first_appear_gap": 0.0,
                "std_gap": 0.0,
                "n_valid": 0,
                "n_unique_answers": 0,
                "last_change_step_frac": 1.0,
                "max_gap_step_frac": 1.0,
                "first_appear_step_frac": 1.0,
            })
            continue
        vote = _exp_only_vote(valid)
        is_correct = 1 if _is_correct(vote, gt) else 0
        gaps = [ev["gap"] for ev in valid]
        first = valid[0]
        # last-change: walk forward tracking parsed_answer flips
        prev = object()
        last_change = valid[0]
        for ev in valid:
            if ev["parsed_answer"] != prev:
                last_change = ev
                prev = ev["parsed_answer"]
        mx = max(valid, key=lambda e: e["gap"])
        mean_gap = sum(gaps) / len(gaps)
        std_gap = (sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)) ** 0.5
        rows.append({
            "sample_index": i,
            "is_correct": is_correct,
            "max_gap": mx["gap"],
            "mean_gap": mean_gap,
            "last_change_gap": last_change["gap"],
            "first_appear_gap": first["gap"],
            "std_gap": std_gap,
            "n_valid": float(len(valid)),
            "n_unique_answers": float(len(set(ev["parsed_answer"] for ev in valid))),
            "last_change_step_frac": last_change["step"] / last_change["total_steps"],
            "max_gap_step_frac": mx["step"] / mx["total_steps"],
            "first_appear_step_frac": first["step"] / first["total_steps"],
        })
    return rows


def _split(rows, frac=VALIDATION_FRAC, seed=SEED):
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    n_val = int(len(rows) * frac)
    val_ids = set(indices[:n_val])
    val = [r for r in rows if r["sample_index"] in val_ids]
    test = [r for r in rows if r["sample_index"] not in val_ids]
    return val, test


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


def _risk_coverage(rows, score_key, ascending_is_low_reliability=False):
    """Sort by score (high = more reliable), evaluate selective accuracy.

    If `ascending_is_low_reliability` is True, lower score = more reliable
    (e.g. for n_unique_answers); the function flips sign internally.

    Returns list of (coverage, selective_acc, n_kept).
    """
    if not rows:
        return []
    if ascending_is_low_reliability:
        ordered = sorted(rows, key=lambda r: r[score_key])
    else:
        ordered = sorted(rows, key=lambda r: -r[score_key])
    n = len(ordered)
    out = []
    for c in COVERAGES:
        k = max(1, int(round(c * n)))
        kept = ordered[:k]
        acc = sum(r["is_correct"] for r in kept) / k
        out.append({"coverage": c, "n_kept": k, "selective_acc": acc})
    return out


def _aurc(curve):
    """Lower-Riemann AURC estimate over the sweep points.

    AURC = mean over coverages of (1 - selective_acc). Lower = better.
    Curve is assumed sorted from high to low coverage.
    """
    if not curve:
        return None
    return sum(1.0 - p["selective_acc"] for p in curve) / len(curve)


def _coverage_at_acc(curve, acc_threshold):
    """Largest coverage at which selective_acc >= threshold; None if never."""
    feasible = [p["coverage"] for p in curve if p["selective_acc"] >= acc_threshold]
    return max(feasible) if feasible else None


def _oracle_curve(rows):
    """Ranking by ground-truth correctness — upper bound."""
    ordered = sorted(rows, key=lambda r: -r["is_correct"])
    n = len(ordered)
    out = []
    for c in COVERAGES:
        k = max(1, int(round(c * n)))
        kept = ordered[:k]
        acc = sum(r["is_correct"] for r in kept) / k
        out.append({"coverage": c, "n_kept": k, "selective_acc": acc})
    return out


def _random_curve(rows, seed=SEED):
    """Random ordering — should hover at base rate."""
    rng = random.Random(seed)
    ordered = list(rows)
    rng.shuffle(ordered)
    n = len(ordered)
    out = []
    for c in COVERAGES:
        k = max(1, int(round(c * n)))
        kept = ordered[:k]
        acc = sum(r["is_correct"] for r in kept) / k
        out.append({"coverage": c, "n_kept": k, "selective_acc": acc})
    return out


def _fit_combined_score(val_rows, features):
    """Fit a val-only z-score combination.

    For each feature:
      sign = sign(Pearson(feature, is_correct))  on val
      mu, sd = mean, std of feature on val

    Combined score = sum_i sign_i * (x_i - mu_i) / sd_i
    """
    spec = {}
    ys = [r["is_correct"] for r in val_rows]
    for f in features:
        xs = [r[f] for r in val_rows]
        mu = sum(xs) / len(xs)
        sd = (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5
        if sd == 0:
            continue
        p = _pearson(xs, ys)
        if abs(p) < 0.05:
            # treat near-zero features as uninformative; drop
            continue
        spec[f] = {"mu": mu, "sd": sd, "sign": 1.0 if p >= 0 else -1.0, "pearson": p}
    return spec


def _apply_combined_score(rows, spec, score_key="combined_score"):
    for r in rows:
        s = 0.0
        for f, p in spec.items():
            z = (r[f] - p["mu"]) / p["sd"]
            s += p["sign"] * z
        r[score_key] = s


def _curves_for_task(rows, feature_corrs):
    """Return per-feature risk-coverage curves auto-oriented by val-corr sign."""
    out = {}
    for f, corr in feature_corrs.items():
        flip = corr < 0
        curve = _risk_coverage(rows, f, ascending_is_low_reliability=flip)
        out[f] = {
            "orient": "ascending" if flip else "descending",
            "val_pearson": corr,
            "curve": curve,
            "aurc": _aurc(curve),
            "cov_at_0_75": _coverage_at_acc(curve, 0.75),
            "cov_at_0_80": _coverage_at_acc(curve, 0.80),
            "cov_at_0_90": _coverage_at_acc(curve, 0.90),
        }
    return out


def _summary_table(curves):
    rows = []
    for f, info in sorted(curves.items(), key=lambda kv: (kv[1]["aurc"] if kv[1]["aurc"] is not None else 1.0)):
        c80 = next((p for p in info["curve"] if abs(p["coverage"] - 0.8) < 1e-6), None)
        c50 = next((p for p in info["curve"] if abs(p["coverage"] - 0.5) < 1e-6), None)
        rows.append({
            "feature": f,
            "val_pearson": info["val_pearson"],
            "aurc": info["aurc"],
            "selacc_at_80": c80["selective_acc"] if c80 else None,
            "selacc_at_50": c50["selective_acc"] if c50 else None,
            "cov_at_90acc": info["cov_at_0_90"],
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gsm8k", default=DEFAULT_GSM8K)
    parser.add_argument("--svamp", default=DEFAULT_SVAMP)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    print(f"[load] GSM8K: {args.gsm8k}")
    gsm_rows = _load(args.gsm8k)
    print(f"  n={len(gsm_rows)}, base_acc={sum(r['is_correct'] for r in gsm_rows)/len(gsm_rows)*100:.2f}%")
    val, test = _split(gsm_rows)
    print(f"[split] val n={len(val)}  test n={len(test)}  (seed={SEED}, frac={VALIDATION_FRAC})")

    # === Single-feature val correlations and per-task curves ===
    ys_val = [r["is_correct"] for r in val]
    val_corrs = {}
    for f in FEATURES:
        xs = [r[f] for r in val]
        val_corrs[f] = _pearson(xs, ys_val)

    print("\n[val Pearson(feature, is_correct)]")
    for f, p in sorted(val_corrs.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {f:25s}  {p:+.3f}")

    gsm_val_curves = _curves_for_task(val, val_corrs)
    gsm_test_curves = _curves_for_task(test, val_corrs)

    # === Combined score (val-fitted, applied to test/SVAMP) ===
    spec = _fit_combined_score(val, FEATURES)
    print(f"\n[combined score spec] using features: {list(spec.keys())}")
    _apply_combined_score(val, spec)
    _apply_combined_score(test, spec)
    combined_val_curve = _risk_coverage(val, "combined_score", ascending_is_low_reliability=False)
    combined_test_curve = _risk_coverage(test, "combined_score", ascending_is_low_reliability=False)

    # === Reference curves ===
    oracle_val = _oracle_curve(val)
    oracle_test = _oracle_curve(test)
    random_val = _random_curve(val)
    random_test = _random_curve(test)

    # === SVAMP transfer ===
    print(f"\n[load] SVAMP: {args.svamp}")
    svamp_rows = _load(args.svamp)
    print(f"  n={len(svamp_rows)}, base_acc={sum(r['is_correct'] for r in svamp_rows)/len(svamp_rows)*100:.2f}%")
    _apply_combined_score(svamp_rows, spec)
    svamp_single_curves = _curves_for_task(svamp_rows, val_corrs)
    svamp_combined_curve = _risk_coverage(svamp_rows, "combined_score", ascending_is_low_reliability=False)
    svamp_oracle = _oracle_curve(svamp_rows)
    svamp_random = _random_curve(svamp_rows)

    # === Output ===
    os.makedirs(args.out_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    stem = f"reliability_abstention_{date_str}"

    def _serialize_curves(curves):
        return {f: {"orient": v["orient"], "val_pearson": v["val_pearson"],
                    "aurc": v["aurc"], "cov_at_0_75": v["cov_at_0_75"],
                    "cov_at_0_80": v["cov_at_0_80"], "cov_at_0_90": v["cov_at_0_90"],
                    "curve": v["curve"]}
                for f, v in curves.items()}

    out = {
        "config": {
            "seed": SEED, "validation_frac": VALIDATION_FRAC, "alpha": ALPHA,
            "coverages": COVERAGES, "features": FEATURES,
        },
        "gsm8k_run": args.gsm8k,
        "svamp_run": args.svamp,
        "n": {"gsm8k": len(gsm_rows), "val": len(val), "test": len(test), "svamp": len(svamp_rows)},
        "base_acc": {
            "gsm8k_full": sum(r["is_correct"] for r in gsm_rows) / len(gsm_rows),
            "val": sum(r["is_correct"] for r in val) / len(val),
            "test": sum(r["is_correct"] for r in test) / len(test),
            "svamp": sum(r["is_correct"] for r in svamp_rows) / len(svamp_rows),
        },
        "val_pearson": val_corrs,
        "combined_score_spec": spec,
        "gsm8k_val": {
            "single_features": _serialize_curves(gsm_val_curves),
            "combined": {"curve": combined_val_curve, "aurc": _aurc(combined_val_curve),
                         "cov_at_0_80": _coverage_at_acc(combined_val_curve, 0.80),
                         "cov_at_0_90": _coverage_at_acc(combined_val_curve, 0.90)},
            "oracle": {"curve": oracle_val, "aurc": _aurc(oracle_val)},
            "random": {"curve": random_val, "aurc": _aurc(random_val)},
        },
        "gsm8k_test": {
            "single_features": _serialize_curves(gsm_test_curves),
            "combined": {"curve": combined_test_curve, "aurc": _aurc(combined_test_curve),
                         "cov_at_0_80": _coverage_at_acc(combined_test_curve, 0.80),
                         "cov_at_0_90": _coverage_at_acc(combined_test_curve, 0.90)},
            "oracle": {"curve": oracle_test, "aurc": _aurc(oracle_test)},
            "random": {"curve": random_test, "aurc": _aurc(random_test)},
        },
        "svamp_transfer": {
            "single_features": _serialize_curves(svamp_single_curves),
            "combined": {"curve": svamp_combined_curve, "aurc": _aurc(svamp_combined_curve),
                         "cov_at_0_80": _coverage_at_acc(svamp_combined_curve, 0.80),
                         "cov_at_0_90": _coverage_at_acc(svamp_combined_curve, 0.90)},
            "oracle": {"curve": svamp_oracle, "aurc": _aurc(svamp_oracle)},
            "random": {"curve": svamp_random, "aurc": _aurc(svamp_random)},
        },
    }

    json_path = os.path.join(args.out_dir, f"{stem}.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote: {json_path}")

    # === Markdown summary ===
    lines = []
    lines.append(f"# Reliability-aware Abstention (Phase E) — {date_str}")
    lines.append("")
    lines.append(f"Source: `{args.gsm8k}`")
    lines.append(f"Transfer: `{args.svamp}`")
    lines.append(f"seed={SEED}; validation_frac={VALIDATION_FRAC}")
    lines.append("")
    lines.append(f"GSM8K n={len(gsm_rows)} (val {len(val)} / test {len(test)}); SVAMP n={len(svamp_rows)}")
    lines.append("")
    lines.append("Base accuracy (no abstention):")
    lines.append(f"- GSM8K full: {out['base_acc']['gsm8k_full']*100:.2f}%")
    lines.append(f"- GSM8K val:  {out['base_acc']['val']*100:.2f}%")
    lines.append(f"- GSM8K test: {out['base_acc']['test']*100:.2f}%")
    lines.append(f"- SVAMP:      {out['base_acc']['svamp']*100:.2f}%")
    lines.append("")

    lines.append("## Single-feature val Pearson")
    lines.append("")
    lines.append("| Feature | Val Pearson |")
    lines.append("|---|---:|")
    for f, p in sorted(val_corrs.items(), key=lambda kv: -abs(kv[1])):
        lines.append(f"| {f} | {p:+.3f} |")
    lines.append("")
    lines.append(f"Combined score features (kept |corr|>=0.05): {list(spec.keys())}")
    lines.append("")

    def _curve_md(curve, n):
        c80 = next((p for p in curve if abs(p["coverage"] - 0.8) < 1e-6), None)
        c50 = next((p for p in curve if abs(p["coverage"] - 0.5) < 1e-6), None)
        return c80["selective_acc"] if c80 else None, c50["selective_acc"] if c50 else None

    def _summary_block(title, single_curves, combined, oracle, random_, n):
        lines.append(f"## {title} (n={n})")
        lines.append("")
        lines.append("| Score | AURC | SelAcc@80% | SelAcc@50% | Cov@90%acc |")
        lines.append("|---|---:|---:|---:|---:|")
        for f, info in sorted(single_curves.items(),
                              key=lambda kv: kv[1]["aurc"] if kv[1]["aurc"] is not None else 1.0):
            a80, a50 = _curve_md(info["curve"], n)
            cov90 = info["cov_at_0_90"]
            cov90_s = f"{cov90*100:.0f}%" if cov90 is not None else "n/a"
            lines.append(f"| {f} | {info['aurc']:.4f} | {a80*100:.2f}% | {a50*100:.2f}% | {cov90_s} |")
        a80, a50 = _curve_md(combined["curve"], n)
        cov90 = combined["cov_at_0_90"]
        cov90_s = f"{cov90*100:.0f}%" if cov90 is not None else "n/a"
        lines.append(f"| **combined** | **{combined['aurc']:.4f}** | **{a80*100:.2f}%** | **{a50*100:.2f}%** | **{cov90_s}** |")
        a80, a50 = _curve_md(oracle["curve"], n)
        lines.append(f"| oracle | {oracle['aurc']:.4f} | {a80*100:.2f}% | {a50*100:.2f}% | — |")
        a80, a50 = _curve_md(random_["curve"], n)
        lines.append(f"| random | {random_['aurc']:.4f} | {a80*100:.2f}% | {a50*100:.2f}% | — |")
        lines.append("")

    _summary_block("GSM8K Validation", gsm_val_curves,
                   out["gsm8k_val"]["combined"], out["gsm8k_val"]["oracle"], out["gsm8k_val"]["random"], len(val))
    _summary_block("GSM8K Test (one-shot)", gsm_test_curves,
                   out["gsm8k_test"]["combined"], out["gsm8k_test"]["oracle"], out["gsm8k_test"]["random"], len(test))
    _summary_block("SVAMP transfer", svamp_single_curves,
                   out["svamp_transfer"]["combined"], out["svamp_transfer"]["oracle"], out["svamp_transfer"]["random"], len(svamp_rows))

    lines.append("## Reading guide")
    lines.append("")
    lines.append("- **AURC**: mean (1 − selective accuracy) across the coverage sweep. Lower = better. `random` and `oracle` bracket the achievable range; a score is only useful if its AURC sits meaningfully closer to oracle than to random.")
    lines.append("- **SelAcc@80%**: selective accuracy when we keep the 80% of samples most reliable by the score (abstain on the bottom 20%). A useful score should lift this above the base accuracy at the same coverage.")
    lines.append("- **Cov@90%acc**: the largest coverage at which selective accuracy stays ≥ 90%. `n/a` means no coverage point in the sweep crosses 90% selective accuracy.")
    lines.append("- Combined score is the val-fitted signed z-score sum, applied unchanged to test and SVAMP — no test/transfer-side refitting.")
    lines.append("")

    md_path = os.path.join(args.out_dir, f"{stem}.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
