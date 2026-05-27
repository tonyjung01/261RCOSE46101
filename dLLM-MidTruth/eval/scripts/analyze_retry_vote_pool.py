"""Cheap analog to Phase E-Retry-Pool: vote-method diversity pool on a
single retry trajectory.

Why this exists. The two retry artifacts we have on the flagged 60
samples (`v6` exp voting and `v1` prob voting) share identical raw
generations (`60/60` text-equal) because temperature=0.0 makes the
trajectory deterministic. They are not two independent regenerations;
they are one trajectory read off two ways. So this analysis is NOT a
true K-seed retry pool — it is a vote-method pool over a single retry.

What it measures. Of the 60 flagged samples, how many are correct under
at least one of the available reads of the same retry trajectory? Each
read is a "pool member":

  - v6 exp_only_answer
  - v6 vote_answer (exp voting on retry)
  - v1 exp_only_answer        (same as v6 exp_only since the trajectory matches)
  - v1 vote_answer (cgap prob voting on retry)

If the trajectory is truly identical, exp_only across artifacts should
match exactly (sanity check); the union upper bound is then capped by
the trajectory itself, and any rescue gain comes from vote-method
choice. A bigger upper bound would need a true K-seed retry pool with
fresh trajectories (Phase E-Retry-Pool proper, requires GPU reruns).

Outputs
  eval/analysis/retry_vote_pool_20260514.{md,json}
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-artifact", default=e4.DEFAULT_GSM8K_BASE)
    parser.add_argument("--prob-artifact", default=e4.DEFAULT_GSM8K_PROB)
    parser.add_argument("--block-artifact", default=e4.DEFAULT_GSM8K_BLOCKACTIVE)
    parser.add_argument("--retry-exp", default=os.path.join(
        REPO_ROOT, "outputs/LLaDA-8B-Instruct/20260514_gsm8k_retry_exp_testsubset_tau05845_v6/rank_0_generations.json"))
    parser.add_argument("--retry-prob", default=os.path.join(
        REPO_ROOT, "outputs/LLaDA-8B-Instruct/20260514_gsm8k_retry_prob_testsubset_tau05845_v1/rank_0_generations.json"))
    parser.add_argument("--score-key", default="logistic_broad_score")
    parser.add_argument("--tau", type=float, default=0.5845)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    rows = e_eval._score_rows(args.base_artifact, args.prob_artifact, args.block_artifact, "test")
    flagged = [r for r in rows if float(r[args.score_key]) <= args.tau]
    print(f"[flagged] {len(flagged)}/{len(rows)} test samples below tau={args.tau}")

    # Load retries; index by question
    retry_exp = {r["question"]: r for r in e4._load_answer_artifact(args.retry_exp)}
    retry_prob = {r["question"]: r for r in e4._load_answer_artifact(args.retry_prob)}

    # Sanity: are raw generations text-equal between v6 and v1?
    with open(args.retry_exp) as f:
        d6 = json.load(f)
    with open(args.retry_prob) as f:
        d1 = json.load(f)
    gen6 = {g["question"]: g["generations"][0] for g in d6["generations"] if g.get("generations")}
    gen1 = {g["question"]: g["generations"][0] for g in d1["generations"] if g.get("generations")}
    text_equal = sum(1 for q in gen6 if q in gen1 and gen6[q] == gen1[q])
    text_diff = sum(1 for q in gen6 if q in gen1 and gen6[q] != gen1[q])
    print(f"[sanity] raw generations text equal: {text_equal}/{text_equal + text_diff}")

    # Per-sample pool reads (on flagged 60)
    per_sample = []
    for r in flagged:
        gt = r["ground_truth"]
        e6 = retry_exp.get(r["question"])
        e1r = retry_prob.get(r["question"])
        reads = {
            "v6_exp_only": e6["exp_only_answer"] if e6 else None,
            "v6_vote": e6["stored_vote_answer"] if e6 else None,
            "v1_exp_only": e1r["exp_only_answer"] if e1r else None,
            "v1_vote": e1r["stored_vote_answer"] if e1r else None,
        }
        correct = {k: int(e1._is_correct(v, gt)) for k, v in reads.items()}
        union_any = int(any(correct.values()))
        intersect_all = int(all(correct.values()))
        per_sample.append({
            "sample_index": r["sample_index"],
            "base_correct": int(e1._is_correct(r["exp_only_answer"], gt)),
            "reads": reads,
            "correct_per_read": correct,
            "union_any": union_any,
            "intersect_all": intersect_all,
        })

    n = len(per_sample)
    accs = {}
    for k in ["v6_exp_only", "v6_vote", "v1_exp_only", "v1_vote"]:
        accs[k] = sum(s["correct_per_read"][k] for s in per_sample) / n
    union_acc = sum(s["union_any"] for s in per_sample) / n
    intersect_acc = sum(s["intersect_all"] for s in per_sample) / n
    base_acc = sum(s["base_correct"] for s in per_sample) / n

    print(f"\n[flagged-60 accuracies]")
    print(f"  base exp_only:          {base_acc*100:.2f}%  ({int(base_acc*n)}/{n})")
    for k, v in accs.items():
        print(f"  retry {k:14s}: {v*100:.2f}%  ({int(v*n)}/{n})")
    print(f"  union-any (oracle pool over reads): {union_acc*100:.2f}%  ({int(union_acc*n)}/{n})")
    print(f"  intersect-all:                      {intersect_acc*100:.2f}%  ({int(intersect_acc*n)}/{n})")

    # Where does vote-method pool rescue beyond a single read?
    rescue_over_v6_exp = []
    for s in per_sample:
        if not s["correct_per_read"]["v6_exp_only"] and s["union_any"]:
            rescue_over_v6_exp.append(s["sample_index"])
    print(f"\n[union-pool rescue beyond v6_exp_only]: {len(rescue_over_v6_exp)} samples ({sorted(rescue_over_v6_exp)})")

    # Compute the full-test policy delta if we used union-pool reads (oracle)
    # on the flagged 60. This is an UPPER BOUND, not deployable (needs an oracle
    # to know which read is right). Still useful to know the ceiling for the
    # vote-pool diversity strategy on this trajectory.
    full_correct = 0
    for r in rows:
        gt = r["ground_truth"]
        is_flagged = float(r[args.score_key]) <= args.tau
        if is_flagged:
            e6 = retry_exp.get(r["question"])
            e1r = retry_prob.get(r["question"])
            cands = [
                r["exp_only_answer"],
                e6["exp_only_answer"] if e6 else None,
                e6["stored_vote_answer"] if e6 else None,
                e1r["exp_only_answer"] if e1r else None,
                e1r["stored_vote_answer"] if e1r else None,
            ]
            if any(e1._is_correct(c, gt) for c in cands):
                full_correct += 1
        else:
            if e1._is_correct(r["exp_only_answer"], gt):
                full_correct += 1
    full_oracle_pool_acc = full_correct / len(rows)
    full_base_acc = sum(int(e1._is_correct(r["exp_only_answer"], r["ground_truth"])) for r in rows) / len(rows)
    print(f"\n[full-test ceiling, if we had an oracle picker over pool reads on flagged 60]")
    print(f"  base:                   {full_base_acc*100:.2f}%")
    print(f"  oracle-pool ceiling:    {full_oracle_pool_acc*100:.2f}%  ({full_correct}/{len(rows)})")

    # === Write ===
    os.makedirs(args.out_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    stem = f"retry_vote_pool_{date_str}"

    out = {
        "config": {"score_key": args.score_key, "tau": args.tau},
        "sources": {
            "retry_exp": args.retry_exp, "retry_prob": args.retry_prob,
        },
        "sanity": {
            "raw_gen_text_equal_v6_v1": text_equal,
            "raw_gen_text_diff_v6_v1": text_diff,
            "interpretation": "60/60 text-equal implies v6 and v1 share the same retry trajectory; this is a vote-method pool, not a regeneration-seed pool",
        },
        "flagged_n": n,
        "flagged_accs": {**accs,
                          "base": base_acc,
                          "union_any": union_acc,
                          "intersect_all": intersect_acc},
        "rescue_over_v6_exp_only": rescue_over_v6_exp,
        "full_test": {
            "base_acc": full_base_acc,
            "oracle_pool_ceiling_on_flagged": full_oracle_pool_acc,
            "n_total": len(rows),
        },
        "per_sample": per_sample,
    }
    json_path = os.path.join(args.out_dir, f"{stem}.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote: {json_path}")

    lines = []
    lines.append(f"# Retry Vote-Method Pool — {date_str}")
    lines.append("")
    lines.append("This is a **cheap analog** to Phase E-Retry-Pool: a vote-method pool over a single retry trajectory. Not a regeneration-seed pool.")
    lines.append("")
    lines.append(f"- retry exp artifact: `{args.retry_exp}`")
    lines.append(f"- retry prob artifact: `{args.retry_prob}`")
    lines.append(f"- raw generation text equality between the two artifacts: `{text_equal}/{text_equal + text_diff}` — the two retry artifacts share the same retry trajectory (temperature=0.0 makes generation deterministic)")
    lines.append(f"- flagged subset on test: `{n}` samples")
    lines.append("")
    lines.append("## Per-read accuracy on the flagged 60")
    lines.append("")
    lines.append("| Read | Correct / 60 | Acc |")
    lines.append("|---|---:|---:|")
    lines.append(f"| base `exp_only` (before retry) | {int(base_acc*n)} | {base_acc*100:.2f}% |")
    for k, v in accs.items():
        lines.append(f"| retry `{k}` | {int(v*n)} | {v*100:.2f}% |")
    lines.append(f"| **union-any** (correct under ≥1 read) | **{int(union_acc*n)}** | **{union_acc*100:.2f}%** |")
    lines.append(f"| intersect-all (correct under all reads) | {int(intersect_acc*n)} | {intersect_acc*100:.2f}% |")
    lines.append("")
    lines.append(f"`union-any − single-read` is the upper bound for vote-method diversity rescue **on this trajectory** (i.e. on this artifact). To get above this ceiling on the flagged samples, fresh regeneration trajectories are needed.")
    lines.append("")
    lines.append("## Full-test ceiling under an oracle pool picker")
    lines.append("")
    lines.append("If we had an oracle that picked the right read out of `{base, v6_exp, v6_vote, v1_exp, v1_vote}` on each flagged sample, full-test accuracy would be:")
    lines.append("")
    lines.append("| Strategy | Full-test acc |")
    lines.append("|---|---:|")
    lines.append(f"| baseline | {full_base_acc*100:.2f}% |")
    lines.append(f"| oracle pool over reads on flagged 60 | {full_oracle_pool_acc*100:.2f}% |")
    lines.append("")
    lines.append("This is a **ceiling**, not deployable — picking the right read requires knowing the answer. Still informative as an upper bound for what vote-method diversity could buy on this single trajectory.")
    lines.append("")
    lines.append("## Reads where the union pool helps beyond `v6_exp_only`")
    lines.append("")
    lines.append(f"`{len(rescue_over_v6_exp)}` flagged samples are correct under at least one read but not under `v6_exp_only`. Sample indices: `{sorted(rescue_over_v6_exp)}`.")
    lines.append("")
    lines.append("## What this means for true E-Retry-Pool")
    lines.append("")
    lines.append("- The two existing retry artifacts (`v6` exp, `v1` prob) share the same retry trajectory (raw generation text equal `60/60`), so they are a vote-method pool, not a regeneration-seed pool.")
    lines.append("- For a real diversity-pool study, fresh retry runs with non-zero temperature or a different sampling seed are needed; the K-seed pool size should aggregate over different trajectories.")
    lines.append("- The vote-method pool union here gives a partial intuition: if even just swapping vote method on a single trajectory rescues additional samples, fresh trajectories should rescue strictly more.")
    md_path = os.path.join(args.out_dir, f"{stem}.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
