"""Option C: gap-as-filter sweep.

Uses existing debug artifacts only (no inference rerun).

Design:
- Split GSM8K 80/20 with the same seed as other offline sweeps.
- For each sample, compute a gap quantile threshold and keep only events whose
  gap is above that threshold.
- Aggregate the surviving events with the original exp temporal weights.
- Compare against a matched-count late-step control to check whether any gain is
  coming from gap ranking rather than simply keeping fewer late events.
- Evaluate the chosen threshold once on the held-out test set and on SVAMP.
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from scripts.sweep_hybrid import (  # noqa: E402
    DEFAULT_ALPHA,
    DEFAULT_GSM8K,
    DEFAULT_OUT_DIR,
    DEFAULT_SVAMP,
    SEED,
    VALIDATION_FRAC,
    _evaluate,
    _is_correct,
    _load_samples,
    _split,
    _vote_with_weights,
)


QUANTILES = [0, 25, 50, 75]


def _filter_events(valid_events, quantile):
    if not valid_events:
        return []
    gaps = np.asarray([ev["gap"] for ev in valid_events], dtype=float)
    threshold = float(np.percentile(gaps, quantile))
    kept = [ev for ev in valid_events if ev["gap"] >= threshold]
    return kept


def _late_keep(valid_events, n_keep):
    if n_keep <= 0:
        return []
    return sorted(valid_events, key=lambda ev: ev["step"], reverse=True)[:n_keep]


def _evaluate_variant(samples, quantile, variant):
    correct = 0
    total = 0
    no_vote = 0
    kept_counts = []
    valid_counts = []

    for sample in samples:
        valid = sample["valid"]
        total += 1
        if not valid:
            no_vote += 1
            continue

        filtered = _filter_events(valid, quantile)
        if variant == "gap_filter":
            kept = filtered
        elif variant == "late_control":
            kept = _late_keep(valid, len(filtered))
        else:
            raise ValueError(f"Unknown variant: {variant}")

        kept_counts.append(len(kept))
        valid_counts.append(len(valid))

        if not kept:
            no_vote += 1
            continue

        vote = _vote_with_weights(kept, lambda ev: ev["exp_weight"])
        if vote is None:
            no_vote += 1
        elif _is_correct(vote, sample["gt"]):
            correct += 1

    mean_keep_ratio = (
        float(np.mean([k / v for k, v in zip(kept_counts, valid_counts)]))
        if kept_counts
        else 0.0
    )
    return {
        "acc": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "n_no_vote": no_vote,
        "mean_keep_ratio": mean_keep_ratio,
        "mean_kept_events": float(np.mean(kept_counts)) if kept_counts else 0.0,
        "mean_valid_events": float(np.mean(valid_counts)) if valid_counts else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gsm8k", default=DEFAULT_GSM8K)
    parser.add_argument("--svamp", default=DEFAULT_SVAMP)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--output-stem", default=None)
    args = parser.parse_args()

    gsm_samples, gsm_meta = _load_samples(args.gsm8k, alpha=args.alpha)
    val, test = _split(gsm_samples, frac=VALIDATION_FRAC, seed=SEED)

    val_exp = _evaluate(val, "exp_only", exp_mode="raw")
    val_cgap = _evaluate(val, "cgap_rawsum")

    val_filter = {}
    val_late = {}
    for q in QUANTILES:
        val_filter[q] = _evaluate_variant(val, q, "gap_filter")
        val_late[q] = _evaluate_variant(val, q, "late_control")

    best_q = max(QUANTILES, key=lambda q: val_filter[q]["acc"])

    test_exp = _evaluate(test, "exp_only", exp_mode="raw")
    test_cgap = _evaluate(test, "cgap_rawsum")
    test_filter = _evaluate_variant(test, best_q, "gap_filter")
    test_late = _evaluate_variant(test, best_q, "late_control")

    svamp_samples, _ = _load_samples(args.svamp, alpha=args.alpha)
    svamp_exp = _evaluate(svamp_samples, "exp_only", exp_mode="raw")
    svamp_cgap = _evaluate(svamp_samples, "cgap_rawsum")
    svamp_filter = _evaluate_variant(svamp_samples, best_q, "gap_filter")
    svamp_late = _evaluate_variant(svamp_samples, best_q, "late_control")

    result = {
        "config": {
            "alpha": args.alpha,
            "seed": SEED,
            "validation_frac": VALIDATION_FRAC,
            "quantiles": QUANTILES,
        },
        "gsm8k_run": args.gsm8k,
        "svamp_run": args.svamp,
        "vote_method": gsm_meta.get("vote_method"),
        "val_baselines": {
            "exp_only": val_exp,
            "cgap_rawsum": val_cgap,
        },
        "val_gap_filter": {str(q): val_filter[q] for q in QUANTILES},
        "val_late_control": {str(q): val_late[q] for q in QUANTILES},
        "best_quantile": best_q,
        "test": {
            "exp_only": test_exp,
            "cgap_rawsum": test_cgap,
            "gap_filter": test_filter,
            "late_control": test_late,
        },
        "transfer_svamp": {
            "exp_only": svamp_exp,
            "cgap_rawsum": svamp_cgap,
            "gap_filter": svamp_filter,
            "late_control": svamp_late,
        },
    }

    os.makedirs(args.out_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    stem = args.output_stem or f"gsm8k_gap_filter_sweep_{date_str}"
    json_path = os.path.join(args.out_dir, f"{stem}.json")
    md_path = os.path.join(args.out_dir, f"{stem}.md")

    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    lines = []
    lines.append(f"# GSM8K Gap-as-Filter Sweep — {date_str}")
    lines.append("")
    lines.append(f"Source: `{args.gsm8k}`")
    lines.append(f"Transfer: `{args.svamp}`")
    lines.append(f"Alpha for exp baseline: {args.alpha}; seed={SEED}; validation_frac={VALIDATION_FRAC}")
    lines.append("")
    lines.append("## Validation baselines")
    lines.append("")
    lines.append("| Method | Val Acc |")
    lines.append("|---|---:|")
    lines.append(f"| exp_only | {val_exp['acc']*100:.2f}% |")
    lines.append(f"| cgap_rawsum | {val_cgap['acc']*100:.2f}% |")
    lines.append("")
    lines.append("## Validation sweep")
    lines.append("")
    lines.append("| Quantile | Gap-filter Val Acc | Late-control Val Acc | Mean keep ratio |")
    lines.append("|---:|---:|---:|---:|")
    for q in QUANTILES:
        gf = val_filter[q]
        lc = val_late[q]
        lines.append(
            f"| {q} | {gf['acc']*100:.2f}% | {lc['acc']*100:.2f}% | {gf['mean_keep_ratio']*100:.1f}% |"
        )
    lines.append("")
    lines.append(f"Best quantile by validation: **{best_q}**")
    lines.append("")
    lines.append("## Test (one-shot, best validation quantile)")
    lines.append("")
    lines.append("| Method | Test Acc | No-vote |")
    lines.append("|---|---:|---:|")
    lines.append(f"| exp_only | {test_exp['acc']*100:.2f}% | {test_exp['n_no_vote']} |")
    lines.append(f"| cgap_rawsum | {test_cgap['acc']*100:.2f}% | {test_cgap['n_no_vote']} |")
    lines.append(f"| gap_filter(q={best_q}) | {test_filter['acc']*100:.2f}% | {test_filter['n_no_vote']} |")
    lines.append(f"| late_control(q={best_q}) | {test_late['acc']*100:.2f}% | {test_late['n_no_vote']} |")
    lines.append("")
    lines.append("## Transfer — SVAMP (same quantile)")
    lines.append("")
    lines.append("| Method | SVAMP Acc |")
    lines.append("|---|---:|")
    lines.append(f"| exp_only | {svamp_exp['acc']*100:.2f}% |")
    lines.append(f"| cgap_rawsum | {svamp_cgap['acc']*100:.2f}% |")
    lines.append(f"| gap_filter(q={best_q}) | {svamp_filter['acc']*100:.2f}% |")
    lines.append(f"| late_control(q={best_q}) | {svamp_late['acc']*100:.2f}% |")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Gap-filter uses gap only to prune events, then reverts to the original exp temporal accumulation on the survivors.")
    lines.append("- Late-control keeps the same number of latest events per sample, which helps check whether any lift is due to gap ranking rather than simple late-step pruning.")
    lines.append("- If gap-filter outperforms both exp_only and matched-count late-control, that is evidence that the gap ranking is more actionable as a gate than as a direct sum weight.")
    lines.append("")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    print(json.dumps(result, indent=2))
    print(f"\nWrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
