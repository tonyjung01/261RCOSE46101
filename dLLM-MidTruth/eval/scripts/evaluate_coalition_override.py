"""Phase 2 B1 — Conservative answer-coalition override rule.

Builds on Phase 1 candidate-set output. Rule:
  default = exp_only
  override iff some alternative answer X has ≥ K temporal-readout support
              [and optionally, sample's logistic_broad_score < tau]
              [and optionally, last_valid_answer also supports X]

Temporal readouts (5):
  last_valid_answer, late_window_majority_q25, late_window_majority_q50,
  longest_run_answer, most_persistent_answer

Sweep K ∈ {2, 3, 4, 5} × { with/without score gate } × { with/without stable filter }.
Val-fit by choosing best config on val; one-shot test evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import analyze_reliability as e1  # noqa: E402
import analyze_reliability_retry as e4  # noqa: E402
import analyze_reliability_v2 as e2  # noqa: E402


TEMPORAL_READOUT_KEYS = [
    "last_valid_answer",
    "late_window_majority_q25",
    "late_window_majority_q50",
    "longest_run_answer",
    "most_persistent_answer",
]


def _normalize(ans):
    if ans is None:
        return None
    try:
        f = float(ans)
        return int(f) if f.is_integer() else f
    except (TypeError, ValueError):
        return str(ans)


def _is_correct(pred, gt, atol=1e-6):
    if pred is None or gt is None:
        return False
    try:
        return abs(float(pred) - float(gt)) < atol
    except (TypeError, ValueError):
        return str(pred) == str(gt)


def _fit_logistic_broad_scores():
    """Reuse the canonical pipeline: fit logistic_broad on GSM8K val, score val+test."""
    gsm_feat = e4._load_feature_rows(e4.DEFAULT_GSM8K_BASE)
    gsm_base_ans = e4._load_answer_artifact(e4.DEFAULT_GSM8K_BASE)
    gsm_prob_ans = e4._load_answer_artifact(e4.DEFAULT_GSM8K_PROB)
    gsm_block_ans = e4._load_answer_artifact(e4.DEFAULT_GSM8K_BLOCKACTIVE)
    merged = e4._merge_sources(gsm_feat, gsm_base_ans, gsm_prob_ans, gsm_block_ans)
    fit = e4._fit_scores(merged)
    score_by_idx = {}
    for s in fit["val"]:
        score_by_idx[s["sample_index"]] = s["logistic_broad_score"]
    for s in fit["test"]:
        score_by_idx[s["sample_index"]] = s["logistic_broad_score"]
    return score_by_idx


def coalition_decision(sample, K, score_tau=None, require_stable=False):
    """Return chosen answer under the coalition rule."""
    candidates = sample["candidates_named"]
    exp_ans = candidates["exp_only"]
    exp_key = _normalize(exp_ans)

    # Count temporal readout support per unique alternative
    alt_support = {}
    for key_name in TEMPORAL_READOUT_KEYS:
        r = candidates[key_name]
        if r is None:
            continue
        rk = _normalize(r)
        if rk == exp_key:
            continue  # supports baseline, not an alternative
        if rk not in alt_support:
            alt_support[rk] = {"answer": r, "readouts": []}
        alt_support[rk]["readouts"].append(key_name)

    if not alt_support:
        return exp_ans, "no_alternative"

    # Pick most-supported alternative
    best_key, best = max(alt_support.items(), key=lambda kv: len(kv[1]["readouts"]))
    if len(best["readouts"]) < K:
        return exp_ans, "insufficient_coalition"

    if score_tau is not None and sample.get("logistic_broad_score") is not None:
        if sample["logistic_broad_score"] >= score_tau:
            return exp_ans, "score_gate_blocks"

    if require_stable:
        # require last_valid_answer to also support the alternative
        last_valid = candidates.get("last_valid_answer")
        if last_valid is None or _normalize(last_valid) != best_key:
            return exp_ans, "stable_filter_blocks"

    return best["answer"], f"coalition_K={K}"


def evaluate_rule(samples, K, score_tau=None, require_stable=False):
    correct = 0
    fix = 0
    hurt = 0
    changed = 0
    reasons = {}
    for s in samples:
        gt = s["ground_truth"]
        base = s["candidates_named"]["exp_only"]
        chosen, reason = coalition_decision(s, K, score_tau, require_stable)
        reasons[reason] = reasons.get(reason, 0) + 1
        is_c = int(_is_correct(chosen, gt))
        base_c = int(_is_correct(base, gt))
        correct += is_c
        if chosen != base:
            changed += 1
            if is_c and not base_c:
                fix += 1
            if (not is_c) and base_c:
                hurt += 1
    n = len(samples)
    return {
        "n": n,
        "correct": correct,
        "acc": correct / n if n else 0,
        "fixes": fix,
        "hurts": hurt,
        "changed": changed,
        "reasons": reasons,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-json", default=os.path.join(REPO_ROOT, "analysis/router_oracle_phase1_20260515.json"))
    parser.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "analysis"))
    parser.add_argument("--output-stem", default="")
    args = parser.parse_args()

    print(f"[load] Phase 1 JSON: {args.phase1_json}")
    with open(args.phase1_json) as f:
        phase1 = json.load(f)
    val_samples = phase1["per_sample_val"]
    test_samples = phase1["per_sample_test"]
    print(f"  val n={len(val_samples)}  test n={len(test_samples)}")

    print(f"[fit] logistic_broad scores on GSM8K val")
    score_by_idx = _fit_logistic_broad_scores()
    for s in val_samples + test_samples:
        s["logistic_broad_score"] = score_by_idx.get(s["sample_index"])

    # Baseline accuracy
    base_val_acc = sum(int(_is_correct(s["candidates_named"]["exp_only"], s["ground_truth"])) for s in val_samples) / len(val_samples)
    base_test_acc = sum(int(_is_correct(s["candidates_named"]["exp_only"], s["ground_truth"])) for s in test_samples) / len(test_samples)
    print(f"[baseline] val exp_only={base_val_acc*100:.2f}%  test exp_only={base_test_acc*100:.2f}%")

    # Sweep configs on val
    K_values = [2, 3, 4, 5]
    score_taus = [None, 0.5, 0.5845]
    stable_options = [False, True]

    print(f"\n[val sweep]")
    print(f"{'K':>3} {'tau':>8} {'stable':>7}   {'val_acc':>9} {'delta':>8} {'changed':>9} {'fixes':>7} {'hurts':>7}")
    val_results = []
    for K in K_values:
        for tau in score_taus:
            for stable in stable_options:
                r = evaluate_rule(val_samples, K, tau, stable)
                val_results.append({"K": K, "tau": tau, "stable": stable, **r})
                tau_s = f"{tau:.3f}" if tau is not None else "—"
                print(f"{K:>3} {tau_s:>8} {str(stable):>7}   "
                      f"{r['acc']*100:>8.2f}% {(r['acc']-base_val_acc)*100:>+7.2f}pt "
                      f"{r['changed']:>9d} {r['fixes']:>7d} {r['hurts']:>7d}")

    # Pick best val config
    best = max(val_results, key=lambda r: r["acc"])
    print(f"\n[val best]  K={best['K']} tau={best['tau']} stable={best['stable']} "
          f"val_acc={best['acc']*100:.2f}% delta={ (best['acc']-base_val_acc)*100:+.2f}pt")

    # One-shot test eval at best val config
    test_r = evaluate_rule(test_samples, best["K"], best["tau"], best["stable"])
    print(f"\n[test one-shot at val-best]")
    print(f"  acc: {test_r['acc']*100:.2f}%  delta_vs_base: {(test_r['acc']-base_test_acc)*100:+.2f}pt")
    print(f"  changed: {test_r['changed']}, fixes: {test_r['fixes']}, hurts: {test_r['hurts']}")
    print(f"  reasons: {test_r['reasons']}")

    # Also try a few diagnostic configs on test (informational)
    diag_configs = [
        (2, None, False),
        (3, None, False),
        (4, None, False),
        (3, 0.5845, False),
        (3, None, True),
        (4, None, True),
    ]
    print(f"\n[test diagnostic configs]")
    print(f"{'K':>3} {'tau':>8} {'stable':>7}   {'test_acc':>9} {'delta':>8} {'changed':>9} {'fixes':>7} {'hurts':>7}")
    diag_test = []
    for K, tau, stable in diag_configs:
        r = evaluate_rule(test_samples, K, tau, stable)
        diag_test.append({"K": K, "tau": tau, "stable": stable, **r})
        tau_s = f"{tau:.3f}" if tau is not None else "—"
        print(f"{K:>3} {tau_s:>8} {str(stable):>7}   "
              f"{r['acc']*100:>8.2f}% {(r['acc']-base_test_acc)*100:>+7.2f}pt "
              f"{r['changed']:>9d} {r['fixes']:>7d} {r['hurts']:>7d}")

    # === Write ===
    os.makedirs(args.out_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    stem = args.output_stem or f"coalition_override_phase2b1_{date_str}"

    out = {
        "config": {
            "phase1_json": args.phase1_json,
            "K_values": K_values, "score_taus": score_taus, "stable_options": stable_options,
        },
        "baseline_val_acc": base_val_acc,
        "baseline_test_acc": base_test_acc,
        "val_sweep": val_results,
        "val_best_config": {"K": best["K"], "tau": best["tau"], "stable": best["stable"],
                            "val_acc": best["acc"], "val_delta": best["acc"]-base_val_acc},
        "test_one_shot_at_val_best": {**test_r, "delta_vs_base": test_r["acc"] - base_test_acc},
        "test_diagnostic_configs": diag_test,
    }
    json_path = os.path.join(args.out_dir, f"{stem}.json")
    md_path = os.path.join(args.out_dir, f"{stem}.md")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote: {json_path}")

    lines = []
    lines.append(f"# Phase 2 B1 — Conservative Answer-Coalition Override — {date_str}")
    lines.append("")
    lines.append("Rule: default `exp_only`; override iff some alternative answer has ≥K temporal-readout support, optionally also requiring `logistic_broad_score < tau` and/or `last_valid_answer` agreement. Val-fit, one-shot test eval.")
    lines.append("")
    lines.append(f"- baseline val acc: `{base_val_acc*100:.2f}%`")
    lines.append(f"- baseline test acc: `{base_test_acc*100:.2f}%`")
    lines.append(f"- temporal readouts (5): {TEMPORAL_READOUT_KEYS}")
    lines.append("")
    lines.append("## Val sweep")
    lines.append("")
    lines.append("| K | tau | stable | val_acc | delta | changed | fixes | hurts |")
    lines.append("|---:|---:|:---:|---:|---:|---:|---:|---:|")
    for r in val_results:
        tau_s = f"{r['tau']:.3f}" if r['tau'] is not None else "—"
        delta = r['acc'] - base_val_acc
        is_best = (r["K"] == best["K"] and r["tau"] == best["tau"] and r["stable"] == best["stable"])
        marker = " ←" if is_best else ""
        lines.append(f"| {r['K']} | {tau_s} | {r['stable']} | {r['acc']*100:.2f}%{marker} | `{delta*100:+.2f}pt` | {r['changed']} | {r['fixes']} | {r['hurts']} |")
    lines.append("")
    lines.append(f"**Val-best config**: K={best['K']}, tau={best['tau']}, stable={best['stable']}, val_acc=`{best['acc']*100:.2f}%`")
    lines.append("")
    lines.append("## Test one-shot at val-best config")
    lines.append("")
    lines.append(f"- test acc: **`{test_r['acc']*100:.2f}%`**")
    lines.append(f"- delta vs exp_only baseline: **`{(test_r['acc']-base_test_acc)*100:+.2f}pt`**")
    lines.append(f"- changed: {test_r['changed']}, fixes: {test_r['fixes']}, hurts: {test_r['hurts']}")
    lines.append(f"- reason distribution: `{test_r['reasons']}`")
    lines.append("")
    lines.append("## Test diagnostic configs (informational, NOT used for selection)")
    lines.append("")
    lines.append("| K | tau | stable | test_acc | delta | changed | fixes | hurts |")
    lines.append("|---:|---:|:---:|---:|---:|---:|---:|---:|")
    for r in diag_test:
        tau_s = f"{r['tau']:.3f}" if r['tau'] is not None else "—"
        delta = r['acc'] - base_test_acc
        lines.append(f"| {r['K']} | {tau_s} | {r['stable']} | {r['acc']*100:.2f}% | `{delta*100:+.2f}pt` | {r['changed']} | {r['fixes']} | {r['hurts']} |")
    lines.append("")
    lines.append("## Decision read")
    lines.append("")
    delta_test = (test_r["acc"] - base_test_acc) * 100
    if delta_test >= 1.0:
        lines.append(f"- **Test lift ≥ +1pt** (`{delta_test:+.2f}pt`) → coalition rule produces a meaningful clean improvement. Worth recording.")
    elif delta_test >= 0.5:
        lines.append(f"- Test lift `{delta_test:+.2f}pt`: small but non-trivial. Borderline; multi-seed control would help confirm.")
    elif delta_test > 0:
        lines.append(f"- Test lift `{delta_test:+.2f}pt`: positive but very small. Likely within noise; raw-accuracy line is effectively saturated.")
    else:
        lines.append(f"- Test lift `{delta_test:+.2f}pt`: ≤ 0. Coalition override does not help. Raw-accuracy line is saturated.")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
