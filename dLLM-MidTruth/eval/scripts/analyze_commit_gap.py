"""Phase 8 (offline proxy): commit-time gap exploration.

Existing debug data does not record per-token commit moments, but the
`parsed_answer` trajectory across steps provides natural proxies for
"commit-equivalent" steps:

- first_appear: first step with non-null parsed_answer
- last_change : last step where parsed_answer changes value
- max_gap    : step with largest `raw_weight` (gap-based selection)
- mean_gap   : mean of `raw_weight` over all valid events (existing cgap_rawsum)

For each proxy we report:
1. Single-event vote accuracy (val/test/SVAMP), to see whether one
   commit-equivalent step's answer rivals the full temporal aggregation.
2. Per-sample Pearson/Spearman of the proxy gap value vs is_correct.

If a proxy's gap shows a clearly higher correlation with correctness than
mean_gap (the diffuse signal), that supports investing in the real GPU
hook (Phase 8 spec). If all proxies remain at parity with mean_gap, the
"commit-time" framing buys nothing and hybrid line stays closed.
"""

import argparse
import json
import math
import os
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


def _is_correct(vote, gt, atol=1e-6):
    if vote is None or gt is None:
        return False
    try:
        return abs(float(vote) - float(gt)) < atol
    except (TypeError, ValueError):
        return str(vote) == str(gt)


def _load(path):
    with open(path) as f:
        data = json.load(f)
    samples = []
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
        samples.append({"sample_index": i, "gt": gt, "valid": valid})
    return samples, data


def _proxy_event(valid, kind):
    """Return the single event chosen by the proxy rule, or None."""
    if not valid:
        return None
    if kind == "first_appear":
        return valid[0]
    if kind == "last_change":
        # last step at which parsed_answer differs from the previous one
        prev = object()
        last = valid[0]
        for ev in valid:
            if ev["parsed_answer"] != prev:
                last = ev
                prev = ev["parsed_answer"]
        return last
    if kind == "max_gap":
        return max(valid, key=lambda e: e["gap"])
    if kind == "min_gap":
        return min(valid, key=lambda e: e["gap"])
    raise ValueError(kind)


def _evaluate_single_event(samples, kind):
    """Vote = parsed_answer at the proxy-selected event (single event per sample)."""
    correct = 0
    total = 0
    no_vote = 0
    for s in samples:
        ev = _proxy_event(s["valid"], kind)
        total += 1
        if ev is None:
            no_vote += 1
            continue
        if _is_correct(ev["parsed_answer"], s["gt"]):
            correct += 1
    return {"acc": correct / total if total else 0.0,
            "correct": correct, "total": total, "n_no_vote": no_vote}


def _evaluate_exp_only(samples):
    """Reuse temporal exp voting as baseline (same as sweep_hybrid.py)."""
    correct = 0
    total = 0
    no_vote = 0
    for s in samples:
        valid = s["valid"]
        total += 1
        if not valid:
            no_vote += 1
            continue
        scores = {}
        for ev in valid:
            scores[ev["parsed_answer"]] = scores.get(ev["parsed_answer"], 0.0) + ev["exp_weight"]
        if not scores:
            no_vote += 1
            continue
        vote = max(scores.items(), key=lambda kv: kv[1])[0]
        if _is_correct(vote, s["gt"]):
            correct += 1
    return {"acc": correct / total if total else 0.0,
            "correct": correct, "total": total, "n_no_vote": no_vote}


def _per_sample_features(samples):
    """Extract per-sample feature dict and is_correct using exp_only vote.

    Using exp_only as the correctness reference keeps the analysis decoupled
    from the proxy-vote variants — we are asking "do the proxy features
    correlate with whether the *baseline* gets this sample right?", which is
    the relevant signal-availability question.
    """
    rows = []
    for s in samples:
        valid = s["valid"]
        if not valid:
            continue
        # baseline correctness via exp_only
        scores = {}
        for ev in valid:
            scores[ev["parsed_answer"]] = scores.get(ev["parsed_answer"], 0.0) + ev["exp_weight"]
        if not scores:
            continue
        baseline_vote = max(scores.items(), key=lambda kv: kv[1])[0]
        is_correct = 1 if _is_correct(baseline_vote, s["gt"]) else 0

        gaps = [ev["gap"] for ev in valid]
        steps = [ev["step"] for ev in valid]
        first = _proxy_event(valid, "first_appear")
        last = _proxy_event(valid, "last_change")
        mx = _proxy_event(valid, "max_gap")
        rows.append({
            "is_correct": is_correct,
            "first_appear_gap": first["gap"],
            "first_appear_step_frac": first["step"] / first["total_steps"],
            "last_change_gap": last["gap"],
            "last_change_step_frac": last["step"] / last["total_steps"],
            "max_gap": mx["gap"],
            "max_gap_step_frac": mx["step"] / mx["total_steps"],
            "mean_gap": sum(gaps) / len(gaps),
            "n_valid": len(valid),
            "n_unique_answers": len(set(ev["parsed_answer"] for ev in valid)),
        })
    return rows


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


def _spearman(xs, ys):
    def ranks(vs):
        idx = sorted(range(len(vs)), key=lambda i: vs[i])
        out = [0.0] * len(vs)
        i = 0
        while i < len(vs):
            j = i
            while j + 1 < len(vs) and vs[idx[j + 1]] == vs[idx[i]]:
                j += 1
            r = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                out[idx[k]] = r
            i = j + 1
        return out
    return _pearson(ranks(xs), ranks(ys))


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


PROXIES = ["first_appear", "last_change", "max_gap"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gsm8k", default=DEFAULT_GSM8K)
    parser.add_argument("--svamp", default=DEFAULT_SVAMP)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    print(f"[load] GSM8K cgap debug: {args.gsm8k}")
    gsm, gsm_meta = _load(args.gsm8k)
    print(f"  n={len(gsm)}  vote_method={gsm_meta.get('vote_method')}")

    val, test = _split(gsm)
    print(f"[split] val n={len(val)}  test n={len(test)}  seed={SEED} frac={VALIDATION_FRAC}")

    # === Single-event proxy votes (val) ===
    val_baseline = _evaluate_exp_only(val)
    print(f"\n[val] exp_only baseline: {val_baseline['acc']*100:.2f}%")
    val_proxy = {kind: _evaluate_single_event(val, kind) for kind in PROXIES}
    for kind, r in val_proxy.items():
        print(f"  proxy={kind:14s}: {r['acc']*100:.2f}%  ({r['correct']}/{r['total']}, no_vote={r['n_no_vote']})")

    # === Test (one-shot at each proxy) ===
    test_baseline = _evaluate_exp_only(test)
    print(f"\n[test] exp_only baseline: {test_baseline['acc']*100:.2f}%")
    test_proxy = {kind: _evaluate_single_event(test, kind) for kind in PROXIES}
    for kind, r in test_proxy.items():
        print(f"  proxy={kind:14s}: {r['acc']*100:.2f}%  ({r['correct']}/{r['total']})")

    # === SVAMP transfer ===
    print(f"\n[load] SVAMP cgap debug: {args.svamp}")
    svamp, svamp_meta = _load(args.svamp)
    svamp_baseline = _evaluate_exp_only(svamp)
    print(f"[SVAMP] exp_only baseline: {svamp_baseline['acc']*100:.2f}%  (n={len(svamp)})")
    svamp_proxy = {kind: _evaluate_single_event(svamp, kind) for kind in PROXIES}
    for kind, r in svamp_proxy.items():
        print(f"  proxy={kind:14s}: {r['acc']*100:.2f}%  ({r['correct']}/{r['total']})")

    # === Per-sample feature correlation (full GSM8K) ===
    print(f"\n[corr] per-sample features vs baseline-correctness (full GSM8K)")
    rows = _per_sample_features(gsm)
    print(f"  n_rows={len(rows)}  correct_rate={sum(r['is_correct'] for r in rows)/len(rows)*100:.2f}%")
    feature_corrs = {}
    for feat in ["first_appear_gap", "last_change_gap", "max_gap", "mean_gap",
                 "first_appear_step_frac", "last_change_step_frac", "max_gap_step_frac",
                 "n_valid", "n_unique_answers"]:
        xs = [r[feat] for r in rows]
        ys = [r["is_correct"] for r in rows]
        p = _pearson(xs, ys)
        s = _spearman(xs, ys)
        feature_corrs[feat] = {"pearson": p, "spearman": s}
        print(f"  {feat:30s}  pearson={p:+.3f}  spearman={s:+.3f}")

    # === Write outputs ===
    os.makedirs(args.out_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    stem = f"gsm8k_commit_gap_proxy_{date_str}"

    out = {
        "config": {"alpha": ALPHA, "seed": SEED, "validation_frac": VALIDATION_FRAC,
                   "proxies": PROXIES},
        "gsm8k_run": args.gsm8k,
        "svamp_run": args.svamp,
        "vote_method": gsm_meta.get("vote_method"),
        "split_sizes": {"val": len(val), "test": len(test)},
        "val": {"exp_only": val_baseline, **{f"proxy_{k}": v for k, v in val_proxy.items()}},
        "test": {"exp_only": test_baseline, **{f"proxy_{k}": v for k, v in test_proxy.items()}},
        "svamp": {"exp_only": svamp_baseline, **{f"proxy_{k}": v for k, v in svamp_proxy.items()}},
        "feature_corrs": feature_corrs,
        "n_corr_rows": len(rows),
    }
    json_path = os.path.join(args.out_dir, f"{stem}.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote: {json_path}")

    lines = []
    lines.append(f"# GSM8K Commit-Gap Proxy (Phase 8 offline) — {date_str}")
    lines.append("")
    lines.append(f"Source: `{args.gsm8k}`")
    lines.append(f"Transfer: `{args.svamp}`")
    lines.append(f"alpha={ALPHA}; seed={SEED}; validation_frac={VALIDATION_FRAC}")
    lines.append("")
    lines.append("## Proxy single-event vote — Validation (n={n})".format(n=len(val)))
    lines.append("")
    lines.append("| Method | Val Acc |")
    lines.append("|---|---:|")
    lines.append(f"| exp_only (baseline) | {val_baseline['acc']*100:.2f}% |")
    for k, r in val_proxy.items():
        lines.append(f"| proxy_{k} | {r['acc']*100:.2f}% |")
    lines.append("")
    lines.append("## Proxy single-event vote — Test (n={n})".format(n=len(test)))
    lines.append("")
    lines.append("| Method | Test Acc |")
    lines.append("|---|---:|")
    lines.append(f"| exp_only (baseline) | {test_baseline['acc']*100:.2f}% |")
    for k, r in test_proxy.items():
        lines.append(f"| proxy_{k} | {r['acc']*100:.2f}% |")
    lines.append("")
    lines.append("## Proxy single-event vote — Transfer SVAMP (n={n})".format(n=len(svamp)))
    lines.append("")
    lines.append("| Method | SVAMP Acc |")
    lines.append("|---|---:|")
    lines.append(f"| exp_only (baseline) | {svamp_baseline['acc']*100:.2f}% |")
    for k, r in svamp_proxy.items():
        lines.append(f"| proxy_{k} | {r['acc']*100:.2f}% |")
    lines.append("")
    lines.append("## Per-sample feature correlation with baseline correctness")
    lines.append("")
    lines.append(f"n={len(rows)}, correct_rate={sum(r['is_correct'] for r in rows)/len(rows)*100:.2f}%")
    lines.append("")
    lines.append("| Feature | Pearson | Spearman |")
    lines.append("|---|---:|---:|")
    for feat, c in feature_corrs.items():
        lines.append(f"| {feat} | {c['pearson']:+.3f} | {c['spearman']:+.3f} |")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- This is an **offline proxy** for Phase 8. True commit-time gap (top1-top2 logit at the moment select_indices commits a token) requires a generate.py hook + GPU rerun.")
    lines.append("- The proxy uses `parsed_answer` trajectory to define commit-equivalent steps: `first_appear` (first non-null parse), `last_change` (settlement), `max_gap` (gap-based pick).")
    lines.append("- A single-event proxy vote that matches or beats exp_only would suggest that one commit-equivalent step carries enough signal — motivating the real hook.")
    lines.append("- Correlation comparison (proxy vs `mean_gap`) checks whether commit-equivalent steps' gaps are more informative than the diffuse temporal mean.")
    md_path = os.path.join(args.out_dir, f"{stem}.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
