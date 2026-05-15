"""Phase E-Retry-CrossTask evaluator (SVAMP).

Same policy comparison as `evaluate_retry_control.py` but operating on
SVAMP rows scored with the GSM8K-val-fitted reliability score:

- P0 baseline: `exp_only` on every SVAMP sample
- P_selective: replace base with SVAMP selective retry on flagged samples
- P_random:    replace base with SVAMP random retry on a budget-matched
               random subset (typically complement-only)

This is the direct test of whether the reliability score targets retry
budget on a different task without any SVAMP-specific tuning.
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
import evaluate_true_retry as e_eval  # noqa: E402
import evaluate_retry_control as ctrl  # noqa: E402
import select_svamp_retry_subset as svamp_sel  # noqa: E402


DEFAULT_OUT_DIR = e4.DEFAULT_OUT_DIR


def _load_scored_svamp():
    score = svamp_sel._fit_score_on_gsm_val()
    rows = svamp_sel._load_svamp_scored(score)
    rows = sorted(rows, key=lambda r: r["sample_index"])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selective-retry-artifact", required=True)
    parser.add_argument("--selective-manifest", required=True)
    parser.add_argument("--random-retry-artifact", required=True)
    parser.add_argument("--random-manifest", required=True)
    parser.add_argument("--answer-kind", choices=["exp_only", "vote", "final"], default="exp_only")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--output-stem", default="")
    args = parser.parse_args()

    print("[load] SVAMP rows + apply GSM8K-val-fitted score")
    rows = _load_scored_svamp()
    base_correct = sum(int(e1._is_correct(r["exp_only_answer"], r["ground_truth"])) for r in rows)
    base_acc = base_correct / len(rows)
    print(f"  n={len(rows)}, base_acc={base_acc*100:.2f}%")

    selective_qs, sel_manifest = ctrl._load_manifest_question_set(args.selective_manifest, rows)
    random_qs, rand_manifest = ctrl._load_manifest_question_set(args.random_manifest, rows)
    overlap_count = len(selective_qs & random_qs)
    print(f"[manifests] selective={len(selective_qs)}  random={len(random_qs)}  overlap={overlap_count}")

    sel_map = e_eval._load_retry_answers(args.selective_retry_artifact)
    rand_map = e_eval._load_retry_answers(args.random_retry_artifact)

    p0 = {"acc": base_acc, "correct": base_correct, "n_total": len(rows)}
    p_sel = ctrl._decompose_policy(rows, selective_qs, sel_map, args.answer_kind)
    p_rand = ctrl._decompose_policy(rows, random_qs, rand_map, args.answer_kind)

    print(f"\n[policies on SVAMP, n={len(rows)}]")
    print(f"  P0 baseline:   {p0['acc']*100:.2f}%  ({p0['correct']}/{p0['n_total']})")
    print(f"  P_selective:   {p_sel['acc']*100:.2f}%  fixes={len(p_sel['fix_indices'])}  hurts={len(p_sel['hurt_indices'])}  changed={len(p_sel['changed_indices'])}")
    print(f"  P_random:      {p_rand['acc']*100:.2f}%  fixes={len(p_rand['fix_indices'])}  hurts={len(p_rand['hurt_indices'])}  changed={len(p_rand['changed_indices'])}")

    sl_sel = p_sel["subset_local"]
    sl_rand = p_rand["subset_local"]
    print(f"\n[subset-local accuracy — retried slice only]")
    print(f"  selective subset base:  {sl_sel['base_acc']*100:.2f}%  ({sl_sel['base_correct']}/{sl_sel['subset_n']})")
    print(f"  selective subset retry: {sl_sel['retry_acc']*100:.2f}%  ({sl_sel['retry_correct']}/{sl_sel['subset_n']})  delta {sl_sel['delta']*100:+.2f}pt")
    print(f"  random   subset base:   {sl_rand['base_acc']*100:.2f}%  ({sl_rand['base_correct']}/{sl_rand['subset_n']})")
    print(f"  random   subset retry:  {sl_rand['retry_acc']*100:.2f}%  ({sl_rand['retry_correct']}/{sl_rand['subset_n']})  delta {sl_rand['delta']*100:+.2f}pt")

    delta_sel = p_sel["acc"] - base_acc
    delta_rand = p_rand["acc"] - base_acc
    delta_targeting = delta_sel - delta_rand
    print(f"\n[deltas vs baseline, full SVAMP]")
    print(f"  selective − base:  {delta_sel*100:+.2f}pt")
    print(f"  random   − base:   {delta_rand*100:+.2f}pt")
    print(f"  targeting marginal (selective − random): {delta_targeting*100:+.2f}pt")

    date_str = datetime.now().strftime("%Y%m%d")
    stem = args.output_stem or f"svamp_retry_control_eval_{date_str}"
    os.makedirs(args.out_dir, exist_ok=True)

    out = {
        "task": "svamp",
        "answer_kind": args.answer_kind,
        "sources": {
            "selective_retry_artifact": args.selective_retry_artifact,
            "selective_manifest": args.selective_manifest,
            "random_retry_artifact": args.random_retry_artifact,
            "random_manifest": args.random_manifest,
        },
        "manifests": {"selective_n": len(selective_qs), "random_n": len(random_qs), "overlap_count": overlap_count},
        "policies": {"baseline": p0, "selective": p_sel, "random": p_rand},
        "deltas": {
            "selective_minus_base": delta_sel,
            "random_minus_base": delta_rand,
            "targeting_marginal_value": delta_targeting,
        },
    }
    json_path = os.path.join(args.out_dir, f"{stem}.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote: {json_path}")

    lines = []
    lines.append(f"# SVAMP Cross-task Retry Control — {date_str}")
    lines.append("")
    lines.append("Cross-task transfer of the GSM8K-val-fitted reliability score and `tau=0.5845`. Same score definition, same threshold, no SVAMP-specific tuning.")
    lines.append("")
    lines.append(f"- n SVAMP samples: `{len(rows)}`")
    lines.append(f"- answer kind on retry: `{args.answer_kind}`")
    lines.append(f"- selective manifest: `{args.selective_manifest}`")
    lines.append(f"- random manifest: `{args.random_manifest}`")
    lines.append(f"- selective retry artifact: `{args.selective_retry_artifact}`")
    lines.append(f"- random retry artifact: `{args.random_retry_artifact}`")
    lines.append(f"- selective n: `{len(selective_qs)}`, random n: `{len(random_qs)}`, overlap: `{overlap_count}`")
    lines.append("")
    lines.append("## Full-SVAMP policies")
    lines.append("")
    lines.append("| Policy | Acc | vs baseline | changed | fixes | hurts |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append(f"| P0 baseline `exp_only` | {p0['acc']*100:.2f}% | — | 0 | 0 | 0 |")
    lines.append(
        f"| P_selective (score-targeted retry) | {p_sel['acc']*100:.2f}% | "
        f"`{delta_sel*100:+.2f}pt` | {len(p_sel['changed_indices'])} | {len(p_sel['fix_indices'])} | {len(p_sel['hurt_indices'])} |"
    )
    lines.append(
        f"| P_random (random-budget retry) | {p_rand['acc']*100:.2f}% | "
        f"`{delta_rand*100:+.2f}pt` | {len(p_rand['changed_indices'])} | {len(p_rand['fix_indices'])} | {len(p_rand['hurt_indices'])} |"
    )
    lines.append("")
    lines.append("## Subset-local accuracy (retried slice only)")
    lines.append("")
    lines.append("| Subset | n | base acc | retry acc | delta |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        f"| selective (score-flagged) | {sl_sel['subset_n']} | "
        f"{sl_sel['base_acc']*100:.2f}% | {sl_sel['retry_acc']*100:.2f}% | `{sl_sel['delta']*100:+.2f}pt` |"
    )
    lines.append(
        f"| random | {sl_rand['subset_n']} | "
        f"{sl_rand['base_acc']*100:.2f}% | {sl_rand['retry_acc']*100:.2f}% | `{sl_rand['delta']*100:+.2f}pt` |"
    )
    lines.append("")
    lines.append("## Targeting marginal value")
    lines.append("")
    lines.append(f"`(P_selective − P_random) on full SVAMP = {delta_targeting*100:+.2f}pt`")
    lines.append("")
    lines.append("Interpretation guide (single seed, single retry artifact per policy, cross-task transfer of the GSM8K-fitted score):")
    lines.append("")
    lines.append("- if `P_selective − P_random > 0` and selective subset-local delta is materially larger than random subset-local delta, the score targeting transfers to SVAMP — score-targeted retry beats random retry on a different task without any SVAMP-specific tuning.")
    lines.append("- if `P_selective − P_random ≈ 0` or negative, the score's targeting effect is GSM8K-specific and does not transfer.")
    lines.append("- single-seed result; should be read as artifact-level alongside the GSM8K multi-seed control (`+3.03pt ± 0.62pt`).")
    md_path = os.path.join(args.out_dir, f"{stem}.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
