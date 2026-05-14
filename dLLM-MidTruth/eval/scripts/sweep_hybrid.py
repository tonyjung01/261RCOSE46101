"""Phase 7: Hybrid sweep over `exp + λ·quality(gap, median)`.

Strictly follows the plan:
- 80/20 sample-level split with fixed seed=42
- Factor 1: quality shape (binary/clipped/tanh) at λ=1.0 (additive), pick best on validation
- Factor 2: λ sweep {0.25,0.5,1.0,2.0,5.0} with best shape (additive), pick best on validation
- Factor 3: formula {additive, multiplicative} with best shape+λ, pick best on validation
- Test set evaluated only once at the end (best config from validation)
- Transfer check on SVAMP using the same best config

Uses the existing cgap debug data — no inference rerun.

This script now also supports a Phase 7B-style exploratory pass:
- exp scaling mode sweep (`raw`, `max_norm`, `step_frac`)
- wider λ sweep
- separate output stem to avoid overwriting the original Phase 7 artifact
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime
from statistics import median

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

DEFAULT_ALPHA = 5.0
SEED = 42
VALIDATION_FRAC = 0.8

QUALITY_SHAPES = {
    "binary": lambda gap, med: 1.0 if gap > med else 0.0,
    "clipped": lambda gap, med: 0.0 if med == 0 else min(max((gap - med) / med, 0.0), 1.0),
    "tanh": lambda gap, med: math.tanh(max(gap - med, 0.0)),
}
LAMBDA_CANDIDATES = [0.25, 0.5, 1.0, 2.0, 5.0]
WIDE_LAMBDA_CANDIDATES = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0]
FORMULAS = ["additive", "multiplicative"]
EXP_MODES = ["raw", "max_norm", "step_frac"]


def _is_correct(vote, gt, atol=1e-6):
    if vote is None or gt is None:
        return False
    try:
        return abs(float(vote) - float(gt)) < atol
    except (TypeError, ValueError):
        return str(vote) == str(gt)


def _load_samples(path, alpha=DEFAULT_ALPHA):
    """Return list of (sample_index, gt, valid_events).

    valid_event dict: {step, total_steps, gap, parsed_answer, exp_weight}
    """
    with open(path) as f:
        data = json.load(f)

    out = []
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
            s = step["step"]
            ts = step["total_steps"]
            valid.append({
                "step": s,
                "total_steps": ts,
                "gap": float(rw),
                "parsed_answer": pa,
                "exp_weight": math.exp(s / ts * alpha),
            })
        out.append({"sample_index": i, "gt": gt, "valid": valid})
    return out, data


def _vote_with_weights(valid_events, weight_fn):
    """Aggregate weights per parsed_answer; return winner (or None)."""
    if not valid_events:
        return None
    scores = {}
    for ev in valid_events:
        w = weight_fn(ev)
        if w is None:
            continue
        key = ev["parsed_answer"]
        scores[key] = scores.get(key, 0.0) + w
    if not scores:
        return None
    return max(scores.items(), key=lambda kv: kv[1])[0]


def _exp_weight(ev, ctx, exp_mode):
    raw = ev["exp_weight"]
    if exp_mode == "raw":
        return raw
    if exp_mode == "max_norm":
        return raw / ctx["max_exp"] if ctx["max_exp"] else 0.0
    if exp_mode == "step_frac":
        return ev["step"] / ev["total_steps"]
    raise ValueError(f"Unknown exp_mode: {exp_mode}")


def _make_hybrid_fn(quality_name, lam, formula, exp_mode="raw"):
    qshape = QUALITY_SHAPES[quality_name]

    def fn(ev, ctx):
        q = qshape(ev["gap"], ctx["median_gap"])
        exp_w = _exp_weight(ev, ctx, exp_mode)
        if formula == "additive":
            return exp_w + lam * q
        if formula == "multiplicative":
            return exp_w * (1.0 + lam * q)
        raise ValueError(f"Unknown formula: {formula}")

    return fn


def _evaluate(samples, weight_kind, **kwargs):
    """weight_kind ∈ {'exp_only', 'cgap_rawsum', 'hybrid'}.

    Returns dict with {acc, correct, total, n_no_vote}.
    """
    correct = 0
    total = 0
    no_vote = 0
    for s in samples:
        valid = s["valid"]
        if not valid:
            no_vote += 1
            total += 1
            continue
        ctx = {
            "median_gap": median(ev["gap"] for ev in valid),
            "max_exp": max(ev["exp_weight"] for ev in valid),
        }
        if weight_kind == "exp_only":
            exp_mode = kwargs.get("exp_mode", "raw")
            weight_fn = lambda ev: _exp_weight(ev, ctx, exp_mode)
        elif weight_kind == "cgap_rawsum":
            weight_fn = lambda ev: ev["gap"]
        elif weight_kind == "hybrid":
            hfn = kwargs["hybrid_fn"]
            weight_fn = lambda ev: hfn(ev, ctx)
        else:
            raise ValueError(f"Unknown weight_kind: {weight_kind}")
        vote = _vote_with_weights(valid, weight_fn)
        if vote is None:
            no_vote += 1
        elif _is_correct(vote, s["gt"]):
            correct += 1
        total += 1
    return {
        "acc": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "n_no_vote": no_vote,
    }


def _split(samples, frac=VALIDATION_FRAC, seed=SEED):
    import random
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    n_val = int(len(samples) * frac)
    val_ids = set(indices[:n_val])
    val = [s for s in samples if s["sample_index"] in val_ids]
    test = [s for s in samples if s["sample_index"] not in val_ids]
    return val, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gsm8k", default=DEFAULT_GSM8K)
    parser.add_argument("--svamp", default=DEFAULT_SVAMP)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--mode", choices=["phase7", "phase7b"], default="phase7")
    parser.add_argument("--output-stem", default=None)
    args = parser.parse_args()

    print(f"[load] GSM8K cgap debug: {args.gsm8k}")
    gsm_samples, gsm_meta = _load_samples(args.gsm8k, alpha=args.alpha)
    print(f"  n_samples={len(gsm_samples)}, vote_method={gsm_meta.get('vote_method')}")

    # Sanity: reproduce cgap rawsum on full GSM8K and compare with stored final accuracy
    full_baseline = _evaluate(gsm_samples, "cgap_rawsum")
    print(f"  reproduced cgap_rawsum acc on full GSM8K: {full_baseline['acc']*100:.2f}% (correct {full_baseline['correct']}/{full_baseline['total']})")

    val, test = _split(gsm_samples)
    print(f"[split] val n={len(val)} test n={len(test)} (seed={SEED}, frac={VALIDATION_FRAC})")

    # Validation baselines
    val_exp = _evaluate(val, "exp_only", exp_mode="raw")
    val_cgap = _evaluate(val, "cgap_rawsum")
    print(f"[val baseline] exp_only={val_exp['acc']*100:.2f}%  cgap_rawsum={val_cgap['acc']*100:.2f}%")

    if args.mode == "phase7":
        # Factor 1: quality shape (λ=1.0, additive)
        print(f"\n[Factor 1] quality shape sweep (λ=1.0, additive)")
        f1_results = {}
        for qname in QUALITY_SHAPES:
            hfn = _make_hybrid_fn(qname, lam=1.0, formula="additive", exp_mode="raw")
            r = _evaluate(val, "hybrid", hybrid_fn=hfn)
            f1_results[qname] = r["acc"]
            print(f"  {qname:8s}: {r['acc']*100:.2f}% ({r['correct']}/{r['total']})")
        best_shape = max(f1_results, key=f1_results.get)
        print(f"  → best shape: {best_shape} ({f1_results[best_shape]*100:.2f}%)")

        # Factor 2: λ sweep (best shape, additive)
        print(f"\n[Factor 2] λ sweep (shape={best_shape}, additive)")
        f2_results = {}
        lambda_candidates = LAMBDA_CANDIDATES
        for lam in lambda_candidates:
            hfn = _make_hybrid_fn(best_shape, lam=lam, formula="additive", exp_mode="raw")
            r = _evaluate(val, "hybrid", hybrid_fn=hfn)
            f2_results[lam] = r["acc"]
            print(f"  λ={lam:>5}: {r['acc']*100:.2f}% ({r['correct']}/{r['total']})")
        best_lambda = max(f2_results, key=f2_results.get)
        print(f"  → best λ: {best_lambda} ({f2_results[best_lambda]*100:.2f}%)")

        # Factor 3: formula (best shape+λ)
        print(f"\n[Factor 3] formula sweep (shape={best_shape}, λ={best_lambda})")
        f3_results = {}
        best_exp_mode = "raw"
        exp_mode_results = None
        for formula in FORMULAS:
            hfn = _make_hybrid_fn(best_shape, lam=best_lambda, formula=formula, exp_mode="raw")
            r = _evaluate(val, "hybrid", hybrid_fn=hfn)
            f3_results[formula] = r["acc"]
            print(f"  {formula:>15}: {r['acc']*100:.2f}% ({r['correct']}/{r['total']})")
        best_formula = max(f3_results, key=f3_results.get)
        print(f"  → best formula: {best_formula} ({f3_results[best_formula]*100:.2f}%)")
    else:
        # Phase 7B: first sweep exp scaling modes using clipped + λ=1 + additive
        print(f"\n[Factor 0 / Phase7B] exp scaling sweep (shape=clipped, λ=1.0, additive)")
        exp_mode_results = {}
        for exp_mode in EXP_MODES:
            hfn = _make_hybrid_fn("clipped", lam=1.0, formula="additive", exp_mode=exp_mode)
            r = _evaluate(val, "hybrid", hybrid_fn=hfn)
            exp_mode_results[exp_mode] = r["acc"]
            print(f"  {exp_mode:10s}: {r['acc']*100:.2f}% ({r['correct']}/{r['total']})")
        best_exp_mode = max(exp_mode_results, key=exp_mode_results.get)
        print(f"  → best exp mode: {best_exp_mode} ({exp_mode_results[best_exp_mode]*100:.2f}%)")

        # Keep clipped for now; prior sweep already showed it best/tied.
        best_shape = "clipped"
        f1_results = {"clipped": exp_mode_results[best_exp_mode]}

        print(f"\n[Factor 1 / Phase7B] wider λ sweep (shape={best_shape}, exp_mode={best_exp_mode}, additive)")
        f2_results = {}
        lambda_candidates = WIDE_LAMBDA_CANDIDATES
        for lam in lambda_candidates:
            hfn = _make_hybrid_fn(best_shape, lam=lam, formula="additive", exp_mode=best_exp_mode)
            r = _evaluate(val, "hybrid", hybrid_fn=hfn)
            f2_results[lam] = r["acc"]
            print(f"  λ={lam:>6}: {r['acc']*100:.2f}% ({r['correct']}/{r['total']})")
        best_lambda = max(f2_results, key=f2_results.get)
        print(f"  → best λ: {best_lambda} ({f2_results[best_lambda]*100:.2f}%)")

        print(f"\n[Factor 2 / Phase7B] formula sweep (shape={best_shape}, exp_mode={best_exp_mode}, λ={best_lambda})")
        f3_results = {}
        for formula in FORMULAS:
            hfn = _make_hybrid_fn(best_shape, lam=best_lambda, formula=formula, exp_mode=best_exp_mode)
            r = _evaluate(val, "hybrid", hybrid_fn=hfn)
            f3_results[formula] = r["acc"]
            print(f"  {formula:>15}: {r['acc']*100:.2f}% ({r['correct']}/{r['total']})")
        best_formula = max(f3_results, key=f3_results.get)
        print(f"  → best formula: {best_formula} ({f3_results[best_formula]*100:.2f}%)")

    # Final config
    best_config = {"quality_shape": best_shape, "lambda": best_lambda, "formula": best_formula, "exp_mode": best_exp_mode}
    print(f"\n[Best config from validation] {best_config}")

    # Test set — single evaluation
    print(f"\n[Test] one-shot evaluation on held-out 20%")
    test_exp = _evaluate(test, "exp_only", exp_mode="raw")
    test_cgap = _evaluate(test, "cgap_rawsum")
    hfn_best = _make_hybrid_fn(best_shape, best_lambda, best_formula, exp_mode=best_exp_mode)
    test_hybrid = _evaluate(test, "hybrid", hybrid_fn=hfn_best)
    print(f"  exp_only:     {test_exp['acc']*100:.2f}% ({test_exp['correct']}/{test_exp['total']})")
    print(f"  cgap_rawsum:  {test_cgap['acc']*100:.2f}% ({test_cgap['correct']}/{test_cgap['total']})")
    print(f"  hybrid(best): {test_hybrid['acc']*100:.2f}% ({test_hybrid['correct']}/{test_hybrid['total']})")

    # Transfer: SVAMP (full)
    print(f"\n[Transfer] SVAMP (full, unpatched logit cgap data)")
    svamp_samples, svamp_meta = _load_samples(args.svamp, alpha=args.alpha)
    svamp_exp = _evaluate(svamp_samples, "exp_only", exp_mode="raw")
    svamp_cgap = _evaluate(svamp_samples, "cgap_rawsum")
    svamp_hybrid = _evaluate(svamp_samples, "hybrid", hybrid_fn=hfn_best)
    print(f"  exp_only:     {svamp_exp['acc']*100:.2f}% ({svamp_exp['correct']}/{svamp_exp['total']})")
    print(f"  cgap_rawsum:  {svamp_cgap['acc']*100:.2f}% ({svamp_cgap['correct']}/{svamp_cgap['total']})")
    print(f"  hybrid(best): {svamp_hybrid['acc']*100:.2f}% ({svamp_hybrid['correct']}/{svamp_hybrid['total']})")

    # Write JSON
    out = {
        "config": {
            "alpha": args.alpha,
            "seed": SEED,
            "validation_frac": VALIDATION_FRAC,
            "mode": args.mode,
            "quality_shapes": list(QUALITY_SHAPES.keys()),
            "exp_modes": EXP_MODES,
            "lambda_candidates": lambda_candidates,
            "formulas": FORMULAS,
        },
        "gsm8k_run": args.gsm8k,
        "svamp_run": args.svamp,
        "gsm8k_full_cgap_rawsum_acc": full_baseline["acc"],
        "split_sizes": {"val": len(val), "test": len(test)},
        "val_baselines": {
            "exp_only": val_exp["acc"],
            "cgap_rawsum": val_cgap["acc"],
        },
        "factor1_quality_shape": f1_results,
        "factor0_exp_mode": exp_mode_results,
        "best_exp_mode": best_exp_mode,
        "best_shape": best_shape,
        "factor2_lambda": {str(k): v for k, v in f2_results.items()},
        "best_lambda": best_lambda,
        "factor3_formula": f3_results,
        "best_formula": best_formula,
        "best_config": best_config,
        "test_acc": {
            "exp_only": test_exp["acc"],
            "cgap_rawsum": test_cgap["acc"],
            "hybrid_best": test_hybrid["acc"],
        },
        "transfer_svamp_full": {
            "exp_only": svamp_exp["acc"],
            "cgap_rawsum": svamp_cgap["acc"],
            "hybrid_best": svamp_hybrid["acc"],
        },
    }

    os.makedirs(args.out_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    stem = args.output_stem or (f"gsm8k_hybrid_sweep_{date_str}" if args.mode == "phase7" else f"gsm8k_hybrid_sweep_v2_{date_str}")
    json_path = os.path.join(args.out_dir, f"{stem}.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote: {json_path}")

    md_path = os.path.join(args.out_dir, f"{stem}.md")
    lines = []
    title = "GSM8K Hybrid One-Factor Sweep" if args.mode == "phase7" else "GSM8K Hybrid Sweep v2"
    lines.append(f"# {title} — {date_str}")
    lines.append("")
    lines.append(f"Source: `{args.gsm8k}`")
    lines.append(f"Transfer: `{args.svamp}`")
    lines.append(f"Alpha for exp baseline: {args.alpha}; seed={SEED}; validation_frac={VALIDATION_FRAC}; mode={args.mode}")
    lines.append("")
    lines.append("## Sanity check")
    lines.append(f"- Reproduced cgap_rawsum on full GSM8K: {full_baseline['acc']*100:.2f}% (tracker number for logit cgap vote: 69.83)")
    lines.append("")
    lines.append("## Validation baselines")
    lines.append("")
    lines.append("| Method | Val Acc |")
    lines.append("|---|---:|")
    lines.append(f"| exp_only | {val_exp['acc']*100:.2f}% |")
    lines.append(f"| cgap_rawsum (logit) | {val_cgap['acc']*100:.2f}% |")
    lines.append("")
    if args.mode == "phase7":
        lines.append("## Factor 1 — Quality shape (λ=1.0, additive)")
        lines.append("")
        lines.append("| Shape | Val Acc |")
        lines.append("|---|---:|")
        for q, a in f1_results.items():
            marker = " ←" if q == best_shape else ""
            lines.append(f"| {q} | {a*100:.2f}%{marker} |")
        lines.append("")
        lines.append(f"Best shape: **{best_shape}**")
        lines.append("")
        factor2_title = f"## Factor 2 — λ sweep (shape={best_shape}, additive)"
    else:
        lines.append("## Factor 0 — Exp scaling mode (shape=clipped, λ=1.0, additive)")
        lines.append("")
        lines.append("| Exp mode | Val Acc |")
        lines.append("|---|---:|")
        for mode_name, acc in exp_mode_results.items():
            marker = " ←" if mode_name == best_exp_mode else ""
            lines.append(f"| {mode_name} | {acc*100:.2f}%{marker} |")
        lines.append("")
        lines.append(f"Best exp mode: **{best_exp_mode}**")
        lines.append("")
        factor2_title = f"## Factor 1 — wider λ sweep (shape={best_shape}, exp_mode={best_exp_mode}, additive)"
    lines.append(factor2_title)
    lines.append("")
    lines.append("| λ | Val Acc |")
    lines.append("|---|---:|")
    for lam in lambda_candidates:
        a = f2_results[lam]
        marker = " ←" if lam == best_lambda else ""
        lines.append(f"| {lam} | {a*100:.2f}%{marker} |")
    lines.append("")
    lines.append(f"Best λ: **{best_lambda}**")
    lines.append("")
    factor3_title = (
        f"## Factor 3 — Formula (shape={best_shape}, λ={best_lambda})"
        if args.mode == "phase7"
        else f"## Factor 2 — Formula (shape={best_shape}, exp_mode={best_exp_mode}, λ={best_lambda})"
    )
    lines.append(factor3_title)
    lines.append("")
    lines.append("| Formula | Val Acc |")
    lines.append("|---|---:|")
    for fname in FORMULAS:
        a = f3_results[fname]
        marker = " ←" if fname == best_formula else ""
        lines.append(f"| {fname} | {a*100:.2f}%{marker} |")
    lines.append("")
    lines.append(f"Best formula: **{best_formula}**")
    lines.append("")
    lines.append("## Best config")
    lines.append("")
    lines.append(f"`(exp_mode, quality_shape, λ, formula) = ({best_exp_mode}, {best_shape}, {best_lambda}, {best_formula})`")
    lines.append("")
    lines.append("## Test (one-shot)")
    lines.append("")
    lines.append("| Method | Test Acc |")
    lines.append("|---|---:|")
    lines.append(f"| exp_only | {test_exp['acc']*100:.2f}% |")
    lines.append(f"| cgap_rawsum (logit) | {test_cgap['acc']*100:.2f}% |")
    lines.append(f"| hybrid(best) | {test_hybrid['acc']*100:.2f}% |")
    lines.append("")
    lines.append("## Transfer check — SVAMP (full, unpatched logit cgap data)")
    lines.append("")
    lines.append("| Method | SVAMP Acc |")
    lines.append("|---|---:|")
    lines.append(f"| exp_only | {svamp_exp['acc']*100:.2f}% |")
    lines.append(f"| cgap_rawsum (logit) | {svamp_cgap['acc']*100:.2f}% |")
    lines.append(f"| hybrid(best) | {svamp_hybrid['acc']*100:.2f}% |")
    if args.mode == "phase7b":
        lines.append("")
        lines.append("## Notes")
        lines.append("")
        lines.append("- This v2 pass explores whether the earlier null result was caused by raw exp scale dominating `quality × λ`.")
        lines.append("- `raw` keeps the original exp dynamic range, `max_norm` divides exp by the sample-local max exp, and `step_frac` uses linear step fraction.")
        lines.append("- If the best config still matches `exp_only` or stays below `cgap_rawsum`, that is evidence that simple additive/multiplicative hybrids remain weak even after scale adjustments.")
    lines.append("")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
