"""Phase E2: better combined-score models for reliability-aware abstention.

Builds on `analyze_reliability.py` by comparing the original signed z-score sum
against simple val-fitted logistic models:

- logistic_top3: uses the three strongest interpretable features
  (`max_gap`, `n_unique_answers`, `last_change_step_frac`)
- logistic_broad: uses the broader feature subset retained by the E baseline
  (`|val Pearson| >= 0.05`)

No external ML dependency is required; logistic regression is fit with NumPy and
an inner val split is used only to choose L2 strength.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from datetime import datetime

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import analyze_reliability as e1  # noqa: E402


DEFAULT_GSM8K = e1.DEFAULT_GSM8K
DEFAULT_SVAMP = e1.DEFAULT_SVAMP
DEFAULT_OUT_DIR = e1.DEFAULT_OUT_DIR

INNER_SEED = 7
INNER_FRAC = 0.8
L2_GRID = [0.0, 0.01, 0.1, 1.0, 5.0, 10.0]
TOP3_FEATURES = ["max_gap", "n_unique_answers", "last_change_step_frac"]


def _inner_split(rows, frac=INNER_FRAC, seed=INNER_SEED):
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    n_train = int(len(rows) * frac)
    train_ids = set(idx[:n_train])
    train = [rows[i] for i in range(len(rows)) if i in train_ids]
    dev = [rows[i] for i in range(len(rows)) if i not in train_ids]
    return train, dev


def _prepare_matrix(rows, features, stats=None):
    x = np.array([[float(r[f]) for f in features] for r in rows], dtype=float)
    if stats is None:
        mu = x.mean(axis=0)
        sd = x.std(axis=0)
        sd[sd == 0.0] = 1.0
        stats = {"mu": mu, "sd": sd}
    xz = (x - stats["mu"]) / stats["sd"]
    y = np.array([float(r["is_correct"]) for r in rows], dtype=float)
    return xz, y, stats


def _fit_logreg_numpy(x, y, l2=1.0, lr=0.1, steps=4000):
    n, d = x.shape
    w = np.zeros(d, dtype=float)
    b = 0.0
    for _ in range(steps):
        z = x @ w + b
        # stable sigmoid
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        err = p - y
        grad_w = (x.T @ err) / n + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return {"w": w, "b": b, "l2": l2}


def _predict_scores(rows, features, stats, model):
    x, _, _ = _prepare_matrix(rows, features, stats=stats)
    z = x @ model["w"] + model["b"]
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    return p.tolist()


def _score_rows(rows, score_key, scores):
    out = [dict(r) for r in rows]
    for r, s in zip(out, scores):
        r[score_key] = float(s)
    return out


def _curve_pack(rows, score_key):
    curve = e1._risk_coverage(rows, score_key, ascending_is_low_reliability=False)
    return {
        "curve": curve,
        "aurc": e1._aurc(curve),
        "cov_at_0_80": e1._coverage_at_acc(curve, 0.80),
        "cov_at_0_90": e1._coverage_at_acc(curve, 0.90),
    }


def _selacc(curve, coverage):
    hit = next((p for p in curve if abs(p["coverage"] - coverage) < 1e-9), None)
    return hit["selective_acc"] if hit else None


def _fit_and_eval_logistic(train_rows, dev_rows, full_val_rows, test_rows, svamp_rows, features):
    x_train, y_train, stats = _prepare_matrix(train_rows, features)
    tuning = []
    best = None
    for l2 in L2_GRID:
        model = _fit_logreg_numpy(x_train, y_train, l2=l2)
        dev_scored = _score_rows(
            dev_rows,
            "model_score",
            _predict_scores(dev_rows, features, stats, model),
        )
        dev_pack = _curve_pack(dev_scored, "model_score")
        record = {"l2": l2, "aurc": dev_pack["aurc"], "selacc80": _selacc(dev_pack["curve"], 0.8)}
        tuning.append(record)
        if best is None or record["aurc"] < best["aurc"]:
            best = record

    x_full, y_full, full_stats = _prepare_matrix(full_val_rows, features)
    final_model = _fit_logreg_numpy(x_full, y_full, l2=best["l2"])

    val_scored = _score_rows(full_val_rows, "model_score", _predict_scores(full_val_rows, features, full_stats, final_model))
    test_scored = _score_rows(test_rows, "model_score", _predict_scores(test_rows, features, full_stats, final_model))
    svamp_scored = _score_rows(svamp_rows, "model_score", _predict_scores(svamp_rows, features, full_stats, final_model))

    return {
        "features": features,
        "tuning": tuning,
        "best_l2": best["l2"],
        "weights": {
            "intercept": float(final_model["b"]),
            "coefficients": {f: float(w) for f, w in zip(features, final_model["w"])},
        },
        "val": _curve_pack(val_scored, "model_score"),
        "test": _curve_pack(test_scored, "model_score"),
        "svamp": _curve_pack(svamp_scored, "model_score"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gsm8k", default=DEFAULT_GSM8K)
    parser.add_argument("--svamp", default=DEFAULT_SVAMP)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    gsm_rows = e1._load(args.gsm8k)
    val, test = e1._split(gsm_rows)
    svamp_rows = e1._load(args.svamp)

    ys_val = [r["is_correct"] for r in val]
    val_corrs = {f: e1._pearson([r[f] for r in val], ys_val) for f in e1.FEATURES}

    # Baseline combined score from Phase E.
    zsum_spec = e1._fit_combined_score(val, e1.FEATURES)
    val_z = [dict(r) for r in val]
    test_z = [dict(r) for r in test]
    svamp_z = [dict(r) for r in svamp_rows]
    e1._apply_combined_score(val_z, zsum_spec)
    e1._apply_combined_score(test_z, zsum_spec)
    e1._apply_combined_score(svamp_z, zsum_spec)
    zsum = {
        "features": list(zsum_spec.keys()),
        "val": _curve_pack(val_z, "combined_score"),
        "test": _curve_pack(test_z, "combined_score"),
        "svamp": _curve_pack(svamp_z, "combined_score"),
    }

    broad_features = list(zsum_spec.keys())
    inner_train, inner_dev = _inner_split(val)

    log_top3 = _fit_and_eval_logistic(inner_train, inner_dev, val, test, svamp_rows, TOP3_FEATURES)
    log_broad = _fit_and_eval_logistic(inner_train, inner_dev, val, test, svamp_rows, broad_features)

    # Best single-feature from Phase E by val AURC
    gsm_val_curves = e1._curves_for_task(val, val_corrs)
    best_single_name, best_single_curve = min(
        gsm_val_curves.items(),
        key=lambda kv: kv[1]["aurc"] if kv[1]["aurc"] is not None else 1.0,
    )
    gsm_test_curves = e1._curves_for_task(test, val_corrs)
    svamp_single_curves = e1._curves_for_task(svamp_rows, val_corrs)
    best_single = {
        "feature": best_single_name,
        "val": best_single_curve,
        "test": gsm_test_curves[best_single_name],
        "svamp": svamp_single_curves[best_single_name],
    }

    oracle_val = {"curve": e1._oracle_curve(val), "aurc": e1._aurc(e1._oracle_curve(val))}
    oracle_test = {"curve": e1._oracle_curve(test), "aurc": e1._aurc(e1._oracle_curve(test))}
    oracle_svamp = {"curve": e1._oracle_curve(svamp_rows), "aurc": e1._aurc(e1._oracle_curve(svamp_rows))}
    random_val = {"curve": e1._random_curve(val), "aurc": e1._aurc(e1._random_curve(val))}
    random_test = {"curve": e1._random_curve(test), "aurc": e1._aurc(e1._random_curve(test))}
    random_svamp = {"curve": e1._random_curve(svamp_rows), "aurc": e1._aurc(e1._random_curve(svamp_rows))}

    date_str = datetime.now().strftime("%Y%m%d")
    stem = f"reliability_abstention_v2_{date_str}"
    os.makedirs(args.out_dir, exist_ok=True)

    out = {
        "config": {
            "seed": e1.SEED,
            "validation_frac": e1.VALIDATION_FRAC,
            "inner_seed": INNER_SEED,
            "inner_frac": INNER_FRAC,
            "l2_grid": L2_GRID,
        },
        "gsm8k_run": args.gsm8k,
        "svamp_run": args.svamp,
        "base_acc": {
            "gsm8k_full": sum(r["is_correct"] for r in gsm_rows) / len(gsm_rows),
            "val": sum(r["is_correct"] for r in val) / len(val),
            "test": sum(r["is_correct"] for r in test) / len(test),
            "svamp": sum(r["is_correct"] for r in svamp_rows) / len(svamp_rows),
        },
        "val_pearson": val_corrs,
        "baseline_zsum": zsum,
        "best_single_feature": best_single,
        "logistic_top3": log_top3,
        "logistic_broad": log_broad,
        "oracle": {"val": oracle_val, "test": oracle_test, "svamp": oracle_svamp},
        "random": {"val": random_val, "test": random_test, "svamp": random_svamp},
    }

    json_path = os.path.join(args.out_dir, f"{stem}.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)

    def _row(label, pack):
        return (
            label,
            pack["aurc"],
            _selacc(pack["curve"], 0.8),
            _selacc(pack["curve"], 0.5),
            pack.get("cov_at_0_90"),
        )

    lines = []
    lines.append(f"# Reliability-aware Abstention E2 — {date_str}")
    lines.append("")
    lines.append(f"Source: `{args.gsm8k}`")
    lines.append(f"Transfer: `{args.svamp}`")
    lines.append(f"outer split seed={e1.SEED}; val_frac={e1.VALIDATION_FRAC}; inner_seed={INNER_SEED}; inner_frac={INNER_FRAC}")
    lines.append("")
    lines.append("## Models")
    lines.append("")
    lines.append(f"- `zsum_combined`: Phase E signed z-score sum baseline")
    lines.append(f"- `best_single`: `{best_single_name}`")
    lines.append(f"- `logistic_top3`: NumPy logistic regression on {TOP3_FEATURES}")
    lines.append(f"- `logistic_broad`: NumPy logistic regression on {broad_features}")
    lines.append("")
    lines.append("## Inner-dev L2 selection")
    lines.append("")
    lines.append("| Model | L2 | Dev AURC | Dev SelAcc@80% |")
    lines.append("|---|---:|---:|---:|")
    for name, model in [("logistic_top3", log_top3), ("logistic_broad", log_broad)]:
        for rec in model["tuning"]:
            mark = " **best**" if rec["l2"] == model["best_l2"] else ""
            lines.append(f"| {name}{mark} | {rec['l2']} | {rec['aurc']:.4f} | {rec['selacc80']*100:.2f}% |")
    lines.append("")

    def _block(title, rows, oracle_pack, random_pack):
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Score | AURC | SelAcc@80% | SelAcc@50% | Cov@90%acc |")
        lines.append("|---|---:|---:|---:|---:|")
        for label, pack in rows:
            cov = pack.get("cov_at_0_90")
            cov_s = f"{cov*100:.0f}%" if cov is not None else "n/a"
            lines.append(f"| {label} | {pack['aurc']:.4f} | {_selacc(pack['curve'],0.8)*100:.2f}% | {_selacc(pack['curve'],0.5)*100:.2f}% | {cov_s} |")
        for label, pack in [("oracle", oracle_pack), ("random", random_pack)]:
            lines.append(f"| {label} | {pack['aurc']:.4f} | {_selacc(pack['curve'],0.8)*100:.2f}% | {_selacc(pack['curve'],0.5)*100:.2f}% | — |")
        lines.append("")

    _block(
        "GSM8K Validation",
        [
            ("zsum_combined", zsum["val"]),
            (f"best_single ({best_single_name})", best_single["val"]),
            ("logistic_top3", log_top3["val"]),
            ("logistic_broad", log_broad["val"]),
        ],
        oracle_val,
        random_val,
    )
    _block(
        "GSM8K Test (one-shot)",
        [
            ("zsum_combined", zsum["test"]),
            (f"best_single ({best_single_name})", best_single["test"]),
            ("logistic_top3", log_top3["test"]),
            ("logistic_broad", log_broad["test"]),
        ],
        oracle_test,
        random_test,
    )
    _block(
        "SVAMP transfer",
        [
            ("zsum_combined", zsum["svamp"]),
            (f"best_single ({best_single_name})", best_single["svamp"]),
            ("logistic_top3", log_top3["svamp"]),
            ("logistic_broad", log_broad["svamp"]),
        ],
        oracle_svamp,
        random_svamp,
    )

    lines.append("## Notes")
    lines.append("")
    lines.append("- Logistic models are trained only on the GSM8K validation split, with L2 chosen on an inner dev split.")
    lines.append("- Test and SVAMP scores see the frozen val-fit model only; no refit is performed outside validation.")
    lines.append("- This pass still targets baseline `exp_only` correctness, so improvements here should be read as sample-level reliability gains rather than voting gains.")

    md_path = os.path.join(args.out_dir, f"{stem}.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
