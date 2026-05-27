"""Phase E5: calibration analysis for reliability scores.

Focuses on the learned probability-like score (`logistic_broad`) because Phase
E2/E2-Math500 already suggested it is the most product-like confidence output.

Outputs
  eval/analysis/reliability_calibration_{date}.json
  eval/analysis/reliability_calibration_{date}.md
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import analyze_reliability as e1  # noqa: E402
import analyze_reliability_v2 as e2  # noqa: E402
import analyze_reliability_math500 as m5  # noqa: E402


DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "analysis")
BIN_EDGES = np.linspace(0.0, 1.0, 11)


def _fit_gsm_models():
    gsm_rows = e1._load(e1.DEFAULT_GSM8K)
    val, test = e1._split(gsm_rows)
    svamp_rows = e1._load(e1.DEFAULT_SVAMP)

    ys_val = [r["is_correct"] for r in val]
    val_corrs = {f: e1._pearson([r[f] for r in val], ys_val) for f in e1.FEATURES}
    zsum_spec = e1._fit_combined_score(val, e1.FEATURES)
    broad_features = list(zsum_spec.keys())

    inner_train, inner_dev = e2._inner_split(val)
    x_train, y_train, stats = e2._prepare_matrix(inner_train, broad_features)
    best = None
    for l2 in e2.L2_GRID:
        model = e2._fit_logreg_numpy(x_train, y_train, l2=l2)
        dev_scored = e2._score_rows(
            inner_dev,
            "model_score",
            e2._predict_scores(inner_dev, broad_features, stats, model),
        )
        aurc = e2._curve_pack(dev_scored, "model_score")["aurc"]
        if best is None or aurc < best["aurc"]:
            best = {"l2": l2, "aurc": aurc}

    x_full, y_full, full_stats = e2._prepare_matrix(val, broad_features)
    final_model = e2._fit_logreg_numpy(x_full, y_full, l2=best["l2"])

    def attach(rows):
        out = [dict(r) for r in rows]
        probs = e2._predict_scores(out, broad_features, full_stats, final_model)
        for r, p in zip(out, probs):
            r["prob"] = float(p)
        return out

    return {
        "features": broad_features,
        "best_l2": best["l2"],
        "val": attach(val),
        "test": attach(test),
        "svamp": attach(svamp_rows),
    }


def _fit_math_models():
    rows = m5._load_math_rows(m5.DEFAULT_MATH)
    val, test = m5._split(rows)
    ys_val = [r["is_correct"] for r in val]
    val_corrs = {f: e1._pearson([r[f] for r in val], ys_val) for f in m5.FEATURES}
    broad_features = [f for f, corr in val_corrs.items() if abs(corr) >= 0.05]
    inner_train, inner_dev = m5._inner_split(val)
    x_train, y_train, stats = m5._prepare_matrix(inner_train, broad_features)
    best = None
    for l2 in m5.L2_GRID:
        model = m5._fit_logreg_numpy(x_train, y_train, l2=l2)
        dev_scores = m5._predict_scores(inner_dev, broad_features, stats, model)
        dev_rows = [dict(r) for r in inner_dev]
        for r, s in zip(dev_rows, dev_scores):
            r["prob"] = float(s)
        aurc = m5._curve_pack(dev_rows, "prob")["aurc"]
        if best is None or aurc < best["aurc"]:
            best = {"l2": l2, "aurc": aurc}

    x_full, y_full, full_stats = m5._prepare_matrix(val, broad_features)
    final_model = m5._fit_logreg_numpy(x_full, y_full, l2=best["l2"])

    def attach(rows_):
        out = [dict(r) for r in rows_]
        probs = m5._predict_scores(out, broad_features, full_stats, final_model)
        for r, p in zip(out, probs):
            r["prob"] = float(p)
        return out

    return {
        "features": broad_features,
        "best_l2": best["l2"],
        "val": attach(val),
        "test": attach(test),
    }


def _ece(rows, n_bins=10):
    if not rows:
        return None
    probs = np.array([r["prob"] for r in rows], dtype=float)
    labels = np.array([r["is_correct"] for r in rows], dtype=float)
    total = len(rows)
    ece = 0.0
    mce = 0.0
    bins = []
    for i in range(n_bins):
        lo = BIN_EDGES[i]
        hi = BIN_EDGES[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        n = int(mask.sum())
        if n == 0:
            bins.append({"bin": i, "lo": float(lo), "hi": float(hi), "n": 0})
            continue
        conf = float(probs[mask].mean())
        acc = float(labels[mask].mean())
        gap = abs(acc - conf)
        ece += (n / total) * gap
        mce = max(mce, gap)
        bins.append(
            {
                "bin": i,
                "lo": float(lo),
                "hi": float(hi),
                "n": n,
                "mean_prob": conf,
                "emp_acc": acc,
                "gap": gap,
            }
        )
    brier = float(np.mean((probs - labels) ** 2))
    return {"ece": float(ece), "mce": float(mce), "brier": brier, "bins": bins}


def _render_block(lines, title, pack):
    lines.append(f"## {title}")
    lines.append("")
    lines.append(f"- ECE: `{pack['ece']:.4f}`")
    lines.append(f"- MCE: `{pack['mce']:.4f}`")
    lines.append(f"- Brier: `{pack['brier']:.4f}`")
    lines.append("")
    lines.append("| Bin | Prob range | n | Mean predicted p | Empirical acc | |acc-p| |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for b in pack["bins"]:
        if b["n"] == 0:
            lines.append(f"| {b['bin']} | [{b['lo']:.1f}, {b['hi']:.1f}{']' if b['hi']==1.0 else ')'} | 0 | — | — | — |")
        else:
            right = "]" if b["hi"] == 1.0 else ")"
            lines.append(
                f"| {b['bin']} | [{b['lo']:.1f}, {b['hi']:.1f}{right} | {b['n']} | {b['mean_prob']:.3f} | {b['emp_acc']:.3f} | {b['gap']:.3f} |"
            )
    lines.append("")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    gsm = _fit_gsm_models()
    math = _fit_math_models()

    out = {
        "config": {"bins": BIN_EDGES.tolist()},
        "gsm8k": {
            "features": gsm["features"],
            "best_l2": gsm["best_l2"],
            "val": _ece(gsm["val"]),
            "test": _ece(gsm["test"]),
            "svamp_transfer": _ece(gsm["svamp"]),
        },
        "math500": {
            "features": math["features"],
            "best_l2": math["best_l2"],
            "val": _ece(math["val"]),
            "test": _ece(math["test"]),
        },
    }

    os.makedirs(args.out_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    json_path = os.path.join(args.out_dir, f"reliability_calibration_{date_str}.json")
    md_path = os.path.join(args.out_dir, f"reliability_calibration_{date_str}.md")

    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    lines = []
    lines.append(f"# Reliability Calibration (Phase E5) — {date_str}")
    lines.append("")
    lines.append("Binned calibration analysis for the learned `logistic_broad` reliability score.")
    lines.append("")
    lines.append("Sources:")
    lines.append(f"- GSM8K base: `{e1.DEFAULT_GSM8K}`")
    lines.append(f"- SVAMP transfer: `{e1.DEFAULT_SVAMP}`")
    lines.append(f"- Math500 base: `{m5.DEFAULT_MATH}`")
    lines.append("")
    lines.append("Interpretation notes:")
    lines.append("- Lower ECE / MCE / Brier are better.")
    lines.append("- SVAMP uses the GSM8K-fitted model unchanged, so this is a transfer calibration readout.")
    lines.append("- Math500 uses its own task-specific learned score because the feature set differs materially.")
    lines.append("")
    _render_block(lines, "GSM8K validation", out["gsm8k"]["val"])
    _render_block(lines, "GSM8K test", out["gsm8k"]["test"])
    _render_block(lines, "SVAMP transfer", out["gsm8k"]["svamp_transfer"])
    _render_block(lines, "Math500 validation", out["math500"]["val"])
    _render_block(lines, "Math500 test", out["math500"]["test"])
    lines.append("## Current read")
    lines.append("")
    lines.append("- This pass is about whether the learned reliability score is calibrated enough to be read as a confidence-like quantity, not just as a ranking signal.")
    lines.append("- If monotonic ranking is good but calibration is loose, threshold-based abstention/retry may still work while direct probability interpretation remains weak.")

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
