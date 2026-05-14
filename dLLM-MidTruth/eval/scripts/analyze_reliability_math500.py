"""Math500 reliability-aware abstention pass.

Task-specific extension of the Phase E reliability track. Unlike GSM8K, Math500
has a large AWNF population and strong Bucket-A parser-side structure, so the
feature set explicitly includes:

- valid / AWNF / parse_failed ratios
- Bucket A / Bucket B ratios
- Bucket-A subtype ratios
- the usual valid-event gap / trajectory features on the surviving valid events

Target label: correctness of an offline `exp_only` reconstruction from the valid
events in the debug artifact, scored with `utils.parsers.is_equiv`.
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

import analyze_reliability as base  # noqa: E402
from analyze_bucket_a import classify_bucket_a  # noqa: E402
from utils.parsers import is_equiv  # noqa: E402
from utils.span_parsers import parse_math_answer_with_span  # noqa: E402


DEFAULT_MATH = os.path.join(
    REPO_ROOT,
    "outputs/LLaDA-8B-Instruct/20260430_cgap_answer_window5_prob_mean_rawsum_bs4_all_debug/"
    "math500_gen128_steps64_vote_confidence_gap_answer_window5_prob_mean_rawsum_bs4/"
    "rank_0_generations.json",
)
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "analysis")

SEED = 42
VALIDATION_FRAC = 0.8
INNER_SEED = 7
INNER_FRAC = 0.8
L2_GRID = [0.0, 0.01, 0.1, 1.0, 5.0, 10.0]
ALPHA = 5.0
TOTAL_STEPS = 64.0

FEATURES = [
    "max_gap",
    "mean_gap",
    "last_change_gap",
    "first_valid_gap",
    "std_gap",
    "n_valid",
    "n_unique_answers",
    "last_change_step_frac",
    "max_gap_step_frac",
    "first_valid_step_frac",
    "valid_ratio",
    "awnf_ratio",
    "parse_failed_ratio",
    "bucket_a_ratio",
    "bucket_b_ratio",
    "cat_boxedboxed_ratio",
    "cat_boxed_like_ratio",
    "cat_no_boxed_no_answer_tag_ratio",
    "cat_no_boxed_but_answer_tag_ratio",
]


def _exp_only_vote(valid):
    if not valid:
        return None
    scores = {}
    for ev in valid:
        scores[ev["parsed_answer"]] = scores.get(ev["parsed_answer"], 0.0) + ev["exp_weight"]
    if not scores:
        return None
    return max(scores.items(), key=lambda kv: kv[1])[0]


def _is_correct(vote, gt):
    try:
        return 1 if is_equiv(vote, gt) else 0
    except Exception:
        return 0


def _load_math_rows(path):
    with open(path) as f:
        data = json.load(f)
    rows = []
    for i, vd in enumerate(data["vote_debug"]):
        gt = data["generations"][i]["ground_truth"]
        valid = []
        awnf = 0
        parse_failed = 0
        bucket_a = 0
        bucket_b = 0
        cat_counts = {
            "boxedboxed_corruption": 0,
            "boxed_like_but_unparseable": 0,
            "no_boxed_no_answer_tag": 0,
            "no_boxed_but_answer_tag": 0,
        }

        for step in vd["steps"]:
            sr = step.get("skip_reason")
            if sr is None:
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
            elif sr == "answer_window_not_found":
                awnf += 1
                text = step.get("parsed_answer", "")
                _, cs, _ = parse_math_answer_with_span(text)
                if cs < 0:
                    bucket_a += 1
                    cat_counts[classify_bucket_a(text)] += 1
                else:
                    bucket_b += 1
            elif sr == "parse_failed":
                parse_failed += 1

        vote = _exp_only_vote(valid)
        is_correct = _is_correct(vote, gt)

        if valid:
            gaps = [ev["gap"] for ev in valid]
            first = valid[0]
            prev = object()
            last_change = valid[0]
            for ev in valid:
                if ev["parsed_answer"] != prev:
                    last_change = ev
                    prev = ev["parsed_answer"]
            mx = max(valid, key=lambda e: e["gap"])
            mean_gap = sum(gaps) / len(gaps)
            std_gap = (sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)) ** 0.5
            first_valid_gap = first["gap"]
            first_valid_step_frac = first["step"] / first["total_steps"]
            last_change_gap = last_change["gap"]
            last_change_step_frac = last_change["step"] / last_change["total_steps"]
            max_gap = mx["gap"]
            max_gap_step_frac = mx["step"] / mx["total_steps"]
            n_valid = float(len(valid))
            n_unique_answers = float(len(set(ev["parsed_answer"] for ev in valid)))
        else:
            mean_gap = std_gap = last_change_gap = first_valid_gap = max_gap = 0.0
            first_valid_step_frac = last_change_step_frac = max_gap_step_frac = 1.0
            n_valid = 0.0
            n_unique_answers = 0.0

        rows.append({
            "sample_index": i,
            "is_correct": is_correct,
            "max_gap": max_gap,
            "mean_gap": mean_gap,
            "last_change_gap": last_change_gap,
            "first_valid_gap": first_valid_gap,
            "std_gap": std_gap,
            "n_valid": n_valid,
            "n_unique_answers": n_unique_answers,
            "last_change_step_frac": last_change_step_frac,
            "max_gap_step_frac": max_gap_step_frac,
            "first_valid_step_frac": first_valid_step_frac,
            "valid_ratio": n_valid / TOTAL_STEPS,
            "awnf_ratio": awnf / TOTAL_STEPS,
            "parse_failed_ratio": parse_failed / TOTAL_STEPS,
            "bucket_a_ratio": bucket_a / TOTAL_STEPS,
            "bucket_b_ratio": bucket_b / TOTAL_STEPS,
            "cat_boxedboxed_ratio": cat_counts["boxedboxed_corruption"] / TOTAL_STEPS,
            "cat_boxed_like_ratio": cat_counts["boxed_like_but_unparseable"] / TOTAL_STEPS,
            "cat_no_boxed_no_answer_tag_ratio": cat_counts["no_boxed_no_answer_tag"] / TOTAL_STEPS,
            "cat_no_boxed_but_answer_tag_ratio": cat_counts["no_boxed_but_answer_tag"] / TOTAL_STEPS,
        })
    return rows


def _split(rows, frac=VALIDATION_FRAC, seed=SEED):
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    n_val = int(len(rows) * frac)
    val_ids = set(idx[:n_val])
    return [r for r in rows if r["sample_index"] in val_ids], [r for r in rows if r["sample_index"] not in val_ids]


def _inner_split(rows, frac=INNER_FRAC, seed=INNER_SEED):
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    n_train = int(len(rows) * frac)
    train_ids = set(idx[:n_train])
    return [rows[i] for i in range(len(rows)) if i in train_ids], [rows[i] for i in range(len(rows)) if i not in train_ids]


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
    curve = base._risk_coverage(rows, score_key, ascending_is_low_reliability=False)
    return {
        "curve": curve,
        "aurc": base._aurc(curve),
        "cov_at_0_80": base._coverage_at_acc(curve, 0.80),
        "cov_at_0_90": base._coverage_at_acc(curve, 0.90),
    }


def _selacc(curve, coverage):
    hit = next((p for p in curve if abs(p["coverage"] - coverage) < 1e-9), None)
    return hit["selective_acc"] if hit else None


def _fit_combined_score(val_rows):
    ys = [r["is_correct"] for r in val_rows]
    spec = {}
    for f in FEATURES:
        xs = [r[f] for r in val_rows]
        mu = sum(xs) / len(xs)
        sd = (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5
        if sd == 0:
            continue
        p = base._pearson(xs, ys)
        if abs(p) < 0.05:
            continue
        spec[f] = {"mu": mu, "sd": sd, "sign": 1.0 if p >= 0 else -1.0, "pearson": p}
    return spec


def _apply_combined_score(rows, spec):
    for r in rows:
        s = 0.0
        for f, p in spec.items():
            z = (r[f] - p["mu"]) / p["sd"]
            s += p["sign"] * z
        r["combined_score"] = s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--math", default=DEFAULT_MATH)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    rows = _load_math_rows(args.math)
    val, test = _split(rows)
    ys_val = [r["is_correct"] for r in val]
    val_corrs = {f: base._pearson([r[f] for r in val], ys_val) for f in FEATURES}

    # single features
    val_single = base._curves_for_task(val, val_corrs)
    test_single = base._curves_for_task(test, val_corrs)
    best_single_name, best_single_curve = min(
        val_single.items(),
        key=lambda kv: kv[1]["aurc"] if kv[1]["aurc"] is not None else 1.0,
    )

    # zsum
    spec = _fit_combined_score(val)
    val_z = [dict(r) for r in val]
    test_z = [dict(r) for r in test]
    _apply_combined_score(val_z, spec)
    _apply_combined_score(test_z, spec)
    zsum = {"features": list(spec.keys()), "val": _curve_pack(val_z, "combined_score"), "test": _curve_pack(test_z, "combined_score")}

    # logistic broad
    broad_features = list(spec.keys())
    train, dev = _inner_split(val)
    x_train, y_train, train_stats = _prepare_matrix(train, broad_features)
    tuning = []
    best = None
    for l2 in L2_GRID:
        model = _fit_logreg_numpy(x_train, y_train, l2=l2)
        dev_scored = _score_rows(dev, "model_score", _predict_scores(dev, broad_features, train_stats, model))
        pack = _curve_pack(dev_scored, "model_score")
        rec = {"l2": l2, "aurc": pack["aurc"], "selacc80": _selacc(pack["curve"], 0.8)}
        tuning.append(rec)
        if best is None or rec["aurc"] < best["aurc"]:
            best = rec

    x_full, y_full, full_stats = _prepare_matrix(val, broad_features)
    final_model = _fit_logreg_numpy(x_full, y_full, l2=best["l2"])
    val_log = _score_rows(val, "model_score", _predict_scores(val, broad_features, full_stats, final_model))
    test_log = _score_rows(test, "model_score", _predict_scores(test, broad_features, full_stats, final_model))
    logistic_broad = {
        "features": broad_features,
        "tuning": tuning,
        "best_l2": best["l2"],
        "weights": {
            "intercept": float(final_model["b"]),
            "coefficients": {f: float(w) for f, w in zip(broad_features, final_model["w"])},
        },
        "val": _curve_pack(val_log, "model_score"),
        "test": _curve_pack(test_log, "model_score"),
    }

    oracle_val = {"curve": base._oracle_curve(val), "aurc": base._aurc(base._oracle_curve(val))}
    oracle_test = {"curve": base._oracle_curve(test), "aurc": base._aurc(base._oracle_curve(test))}
    random_val = {"curve": base._random_curve(val), "aurc": base._aurc(base._random_curve(val))}
    random_test = {"curve": base._random_curve(test), "aurc": base._aurc(base._random_curve(test))}

    date_str = datetime.now().strftime("%Y%m%d")
    stem = f"math500_reliability_abstention_{date_str}"
    os.makedirs(args.out_dir, exist_ok=True)

    out = {
        "config": {
            "seed": SEED,
            "validation_frac": VALIDATION_FRAC,
            "inner_seed": INNER_SEED,
            "inner_frac": INNER_FRAC,
            "l2_grid": L2_GRID,
            "features": FEATURES,
        },
        "math_run": args.math,
        "n": {"full": len(rows), "val": len(val), "test": len(test)},
        "base_acc": {
            "full": sum(r["is_correct"] for r in rows) / len(rows),
            "val": sum(r["is_correct"] for r in val) / len(val),
            "test": sum(r["is_correct"] for r in test) / len(test),
        },
        "val_pearson": val_corrs,
        "baseline_zsum": zsum,
        "best_single_feature": {
            "feature": best_single_name,
            "val": best_single_curve,
            "test": test_single[best_single_name],
        },
        "logistic_broad": logistic_broad,
        "oracle": {"val": oracle_val, "test": oracle_test},
        "random": {"val": random_val, "test": random_test},
    }

    json_path = os.path.join(args.out_dir, f"{stem}.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)

    lines = []
    lines.append(f"# Math500 Reliability-aware Abstention — {date_str}")
    lines.append("")
    lines.append(f"Source: `{args.math}`")
    lines.append(f"outer split seed={SEED}; val_frac={VALIDATION_FRAC}; inner_seed={INNER_SEED}; inner_frac={INNER_FRAC}")
    lines.append("")
    lines.append(f"Math500 n={len(rows)} (val {len(val)} / test {len(test)})")
    lines.append("")
    lines.append("Base accuracy (no abstention):")
    lines.append(f"- Math500 full: {out['base_acc']['full']*100:.2f}%")
    lines.append(f"- Math500 val:  {out['base_acc']['val']*100:.2f}%")
    lines.append(f"- Math500 test: {out['base_acc']['test']*100:.2f}%")
    lines.append("")
    lines.append("## Single-feature val Pearson")
    lines.append("")
    lines.append("| Feature | Val Pearson |")
    lines.append("|---|---:|")
    for f, p in sorted(val_corrs.items(), key=lambda kv: -abs(kv[1])):
        lines.append(f"| {f} | {p:+.3f} |")
    lines.append("")
    lines.append(f"Z-score combined features (kept |corr|>=0.05): {list(spec.keys())}")
    lines.append("")

    def _block(title, rows_pack, oracle_pack, random_pack):
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Score | AURC | SelAcc@80% | SelAcc@50% | Cov@90%acc |")
        lines.append("|---|---:|---:|---:|---:|")
        for label, pack in rows_pack:
            cov = pack.get("cov_at_0_90")
            cov_s = f"{cov*100:.0f}%" if cov is not None else "n/a"
            lines.append(f"| {label} | {pack['aurc']:.4f} | {_selacc(pack['curve'],0.8)*100:.2f}% | {_selacc(pack['curve'],0.5)*100:.2f}% | {cov_s} |")
        lines.append(f"| oracle | {oracle_pack['aurc']:.4f} | {_selacc(oracle_pack['curve'],0.8)*100:.2f}% | {_selacc(oracle_pack['curve'],0.5)*100:.2f}% | — |")
        lines.append(f"| random | {random_pack['aurc']:.4f} | {_selacc(random_pack['curve'],0.8)*100:.2f}% | {_selacc(random_pack['curve'],0.5)*100:.2f}% | — |")
        lines.append("")

    _block(
        "Math500 Validation",
        [
            ("zsum_combined", zsum["val"]),
            (f"best_single ({best_single_name})", best_single_curve),
            ("logistic_broad", logistic_broad["val"]),
        ],
        oracle_val,
        random_val,
    )
    _block(
        "Math500 Test (one-shot)",
        [
            ("zsum_combined", zsum["test"]),
            (f"best_single ({best_single_name})", test_single[best_single_name]),
            ("logistic_broad", logistic_broad["test"]),
        ],
        oracle_test,
        random_test,
    )

    lines.append("## Notes")
    lines.append("")
    lines.append("- Target label is offline `exp_only` correctness reconstructed from the valid events in this artifact, scored with `utils.parsers.is_equiv`.")
    lines.append("- Feature set is Math500-specific: it includes AWNF / Bucket-A structure in addition to valid-event gap statistics.")
    lines.append("- This should be read as a task-specific reliability pass, not as a direct continuation of the voting/hybrid line.")

    md_path = os.path.join(args.out_dir, f"{stem}.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
