"""Phase E-Retry-Control evaluation.

Compares three policies on GSM8K test (seed=42 outer split):

- P0: no retry, baseline `exp_only`
- P_selective: score-targeted retry — keep base everywhere except the
  60 flagged samples (`logistic_broad_score <= 0.5845`), which are
  replaced with answers from the score-selected retry artifact
- P_random: random-budget retry — keep base everywhere except a
  manifest-defined random subset of 60 samples, which are replaced
  with answers from the random retry artifact

The head-to-head P_selective vs P_random is the direct measurement of
the reliability score's targeting value for retry budget allocation.

This script does NOT generate retry answers; it consumes existing retry
artifacts. The expected pipeline:

  1. select_retry_subset.py        -> txt manifest of flagged samples
  2. select_random_retry_subset.py -> txt manifest of random samples
  3. eval.py / generate.py reruns the model on each manifest -> retry artifacts
  4. evaluate_retry_control.py     -> this script, compares all three policies

Outputs
  eval/analysis/retry_control_eval_{date}.{md,json}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import scripts.analyze_reliability as e1  # noqa: E402
import scripts.analyze_reliability_retry as e4  # noqa: E402
import scripts.evaluate_true_retry as e_eval  # noqa: E402


DEFAULT_OUT_DIR = e4.DEFAULT_OUT_DIR
DEFAULT_SCORE_KEY = "logistic_broad_score"
DEFAULT_TAU = 0.5845


def _decompose_policy(rows, member_question_set, retry_map, answer_kind):
    """Apply a policy: replace base with retry answer iff question ∈ member set
    AND retry artifact has an answer for that question. Otherwise keep base.

    Returns full-test accuracy plus subset-local statistics on the retried slice.
    """
    correct = 0
    fix_indices = []
    hurt_indices = []
    changed_indices = []
    retry_missing = 0

    # subset-local accumulators: only over rows whose question is in member_question_set
    subset_n = 0
    subset_base_correct = 0
    subset_retry_correct = 0

    for r in rows:
        base = r["exp_only_answer"]
        gt = r["ground_truth"]
        in_member = r["question"] in member_question_set
        use = base
        retry_ans = None
        if in_member:
            subset_n += 1
            subset_base_correct += int(e1._is_correct(base, gt))
            retry_row = retry_map.get(r["question"])
            if retry_row is None:
                retry_missing += 1
            else:
                if answer_kind == "exp_only":
                    retry_ans = retry_row["exp_only_answer"]
                elif answer_kind == "vote":
                    retry_ans = retry_row["stored_vote_answer"]
                elif answer_kind == "final":
                    retry_ans = retry_row["stored_final_answer"]
                else:
                    raise ValueError(answer_kind)
                if retry_ans is None:
                    retry_ans = base
                use = retry_ans
                subset_retry_correct += int(e1._is_correct(retry_ans, gt))

        if use != base:
            changed_indices.append(r["sample_index"])
            if e1._is_correct(use, gt) and not e1._is_correct(base, gt):
                fix_indices.append(r["sample_index"])
            if not e1._is_correct(use, gt) and e1._is_correct(base, gt):
                hurt_indices.append(r["sample_index"])
        if e1._is_correct(use, gt):
            correct += 1

    return {
        "acc": correct / len(rows) if rows else 0.0,
        "correct": correct,
        "n_total": len(rows),
        "changed_indices": changed_indices,
        "fix_indices": fix_indices,
        "hurt_indices": hurt_indices,
        "retry_missing": retry_missing,
        "subset_local": {
            "subset_n": subset_n,
            "base_correct": subset_base_correct,
            "base_acc": subset_base_correct / subset_n if subset_n else 0.0,
            "retry_correct": subset_retry_correct,
            "retry_acc": subset_retry_correct / subset_n if subset_n else 0.0,
            "delta": (subset_retry_correct - subset_base_correct) / subset_n if subset_n else 0.0,
        },
    }


def _load_manifest_question_set(manifest_json_path, rows):
    """The manifest stores sample_indices; we map those to question strings
    using the base test rows (1:1)."""
    with open(manifest_json_path) as f:
        manifest = json.load(f)
    wanted = set(int(i) for i in manifest["sample_indices"])
    qs = {r["question"] for r in rows if r["sample_index"] in wanted}
    return qs, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-artifact", default=e4.DEFAULT_GSM8K_BASE)
    parser.add_argument("--prob-artifact", default=e4.DEFAULT_GSM8K_PROB)
    parser.add_argument("--block-artifact", default=e4.DEFAULT_GSM8K_BLOCKACTIVE)
    parser.add_argument("--score-key", default=DEFAULT_SCORE_KEY)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--selective-retry-artifact", required=True,
                        help="retry generations.json from the score-flagged GPU run (e.g. v6)")
    parser.add_argument("--selective-manifest", required=True,
                        help="manifest json from select_retry_subset.py")
    parser.add_argument("--random-retry-artifact", required=True,
                        help="retry generations.json from the random-subset GPU run")
    parser.add_argument("--random-manifest", required=True,
                        help="manifest json from select_random_retry_subset.py")
    parser.add_argument("--answer-kind", choices=["exp_only", "vote", "final"], default="exp_only")
    parser.add_argument("--eval-split", choices=["val", "test", "full"], default="test")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--output-stem", default="")
    args = parser.parse_args()

    print(f"[load] rows on split={args.eval_split}")
    rows = e_eval._score_rows(args.base_artifact, args.prob_artifact, args.block_artifact, args.eval_split)
    print(f"  n={len(rows)}, base_acc={sum(int(e1._is_correct(r['exp_only_answer'], r['ground_truth'])) for r in rows)/len(rows)*100:.2f}%")

    selective_qs, sel_manifest = _load_manifest_question_set(args.selective_manifest, rows)
    random_qs, rand_manifest = _load_manifest_question_set(args.random_manifest, rows)
    overlap_count = len(selective_qs & random_qs)
    print(f"[manifests] selective={len(selective_qs)}  random={len(random_qs)}  overlap={overlap_count}")

    sel_map = e_eval._load_retry_answers(args.selective_retry_artifact)
    rand_map = e_eval._load_retry_answers(args.random_retry_artifact)

    # P0 baseline
    base_correct = sum(int(e1._is_correct(r["exp_only_answer"], r["ground_truth"])) for r in rows)
    p0 = {"acc": base_correct / len(rows), "correct": base_correct, "n_total": len(rows)}

    # P_selective
    p_sel = _decompose_policy(rows, selective_qs, sel_map, args.answer_kind)
    # P_random
    p_rand = _decompose_policy(rows, random_qs, rand_map, args.answer_kind)

    print(f"\n[policies on {args.eval_split}, n={len(rows)}]")
    print(f"  P0 baseline:   {p0['acc']*100:.2f}%  ({p0['correct']}/{p0['n_total']})")
    print(f"  P_selective:   {p_sel['acc']*100:.2f}%  fixes={len(p_sel['fix_indices'])}  hurts={len(p_sel['hurt_indices'])}  changed={len(p_sel['changed_indices'])}")
    print(f"  P_random:      {p_rand['acc']*100:.2f}%  fixes={len(p_rand['fix_indices'])}  hurts={len(p_rand['hurt_indices'])}  changed={len(p_rand['changed_indices'])}")

    # Subset-local stats: accuracy on just the retried slice
    sl_sel = p_sel["subset_local"]
    sl_rand = p_rand["subset_local"]
    print(f"\n[subset-local accuracy — only the {sl_sel['subset_n']}/{sl_rand['subset_n']} retried samples]")
    print(f"  selective subset base:  {sl_sel['base_acc']*100:.2f}%  ({sl_sel['base_correct']}/{sl_sel['subset_n']})")
    print(f"  selective subset retry: {sl_sel['retry_acc']*100:.2f}%  ({sl_sel['retry_correct']}/{sl_sel['subset_n']})  delta {sl_sel['delta']*100:+.2f}pt")
    print(f"  random   subset base:   {sl_rand['base_acc']*100:.2f}%  ({sl_rand['base_correct']}/{sl_rand['subset_n']})")
    print(f"  random   subset retry:  {sl_rand['retry_acc']*100:.2f}%  ({sl_rand['retry_correct']}/{sl_rand['subset_n']})  delta {sl_rand['delta']*100:+.2f}pt")

    # Targeting marginal value
    delta_sel = p_sel["acc"] - p0["acc"]
    delta_rand = p_rand["acc"] - p0["acc"]
    delta_targeting = delta_sel - delta_rand

    print(f"\n[deltas vs baseline, full test]")
    print(f"  selective − base:  {delta_sel*100:+.2f}pt")
    print(f"  random   − base:   {delta_rand*100:+.2f}pt")
    print(f"  targeting marginal value (selective − random): {delta_targeting*100:+.2f}pt")

    # === Output ===
    date_str = datetime.now().strftime("%Y%m%d")
    stem = args.output_stem or f"retry_control_eval_{date_str}"
    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, f"{stem}.json")
    md_path = os.path.join(args.out_dir, f"{stem}.md")

    out = {
        "config": {
            "score_key": args.score_key,
            "tau": args.tau,
            "answer_kind": args.answer_kind,
            "eval_split": args.eval_split,
        },
        "sources": {
            "base_artifact": args.base_artifact,
            "selective_retry_artifact": args.selective_retry_artifact,
            "selective_manifest": args.selective_manifest,
            "random_retry_artifact": args.random_retry_artifact,
            "random_manifest": args.random_manifest,
        },
        "manifests": {
            "selective_n": len(selective_qs),
            "random_n": len(random_qs),
            "overlap_count": overlap_count,
        },
        "policies": {
            "baseline": p0,
            "selective": p_sel,
            "random": p_rand,
        },
        "deltas": {
            "selective_minus_base": delta_sel,
            "random_minus_base": delta_rand,
            "targeting_marginal_value": delta_targeting,
        },
    }
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote: {json_path}")

    lines = []
    lines.append(f"# Phase E-Retry-Control — Selective vs Random Retry — {date_str}")
    lines.append("")
    lines.append(f"- eval split: `{args.eval_split}` (n=`{len(rows)}`)")
    lines.append(f"- answer kind on retry: `{args.answer_kind}`")
    lines.append(f"- score key / tau: `{args.score_key}` / `{args.tau}`")
    lines.append("")
    lines.append(f"- selective manifest: `{args.selective_manifest}` (n_selected=`{len(selective_qs)}`)")
    lines.append(f"- selective retry artifact: `{args.selective_retry_artifact}`")
    lines.append(f"- random manifest: `{args.random_manifest}` (n_selected=`{len(random_qs)}`)")
    lines.append(f"- random retry artifact: `{args.random_retry_artifact}`")
    lines.append(f"- overlap between selective and random subsets: `{overlap_count}` samples")
    lines.append("")
    lines.append("## Full-test policies (single seed, current artifact)")
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
    lines.append("## Subset-local accuracy (only the retried samples)")
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
    lines.append("Subset-local rows show where the retry budget was actually spent. The selective subset's base acc shows how much headroom the score-flagged slice had; the random subset's base acc should be close to overall test accuracy. Comparing the two `delta`s shows whether retry produced more lift on the score-flagged slice than on a random slice of the same size.")
    lines.append("")
    lines.append("## Targeting marginal value")
    lines.append("")
    lines.append(f"`(P_selective − P_random) on full test = {delta_targeting*100:+.2f}pt`")
    lines.append("")
    lines.append("Interpretation guide (single seed, single retry artifact per policy):")
    lines.append("")
    lines.append("- if `P_selective − P_random > 0` and the subset-local delta on the selective slice is materially larger than on the random slice, the reliability score is doing real targeting for retry budget allocation on this artifact; this is the cleanest first-look evidence we have so far.")
    lines.append("- if `P_selective − P_random ≈ 0`, retry helps in general but the score is no better than random at choosing where to spend the budget on this artifact.")
    lines.append("- if `P_selective − P_random < 0`, on this artifact the score is misallocating budget.")
    lines.append("- single-seed measurement: magnitudes are noisy and the sign itself is provisional. The control still gives the cleanest first read on targeting value, but a clean operational claim still wants multi-seed and/or a complement-only random control.")
    lines.append("")
    lines.append("## Overlap caveat")
    lines.append("")
    lines.append("Pure random sampling can overlap with the score-flagged set. Any samples in both manifests are evaluated under each policy independently (i.e., the same sample can be a retry target for both selective and random), so the comparison stays apples-to-apples on the policy effect.")
    lines.append("A stricter optional follow-up is a complement-only random control, drawing the random subset from the score-unflagged samples only (`select_random_retry_subset.py --exclude-flagged`); that disentangles 'score targeting' from 'random sample happened to land on flagged samples.'")
    lines.append("")
    lines.append("## Fix/hurt sample sets")
    lines.append("")
    lines.append(f"- selective fixes: `{p_sel['fix_indices']}`")
    lines.append(f"- selective hurts: `{p_sel['hurt_indices']}`")
    lines.append(f"- random fixes:    `{p_rand['fix_indices']}`")
    lines.append(f"- random hurts:    `{p_rand['hurt_indices']}`")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
