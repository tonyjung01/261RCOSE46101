"""Alternative aggregators on existing K-pool retry artifacts.

User-prioritized aggregator menu (no new GPU):

  (d1) best_of_K_by_margin     : pick the K-retry with the largest
                                  (top1_score - top2_score) margin
  (d2) best_of_K_by_top1       : pick the K-retry with the largest top1_score
  (a)  confidence_weighted_vote: sum each candidate's top1_score across K retries,
                                  pick the candidate with the largest summed score
  (bc) two_of_three_else_base  : if ≥2 of K retries agree on an answer, use that;
                                  else fall back to the original base exp_only answer
                                  (at K=3 this is the union of "base-anchored majority"
                                   and "any-2 agreement")

Plus references:
  K=1                          : seed=42 retry alone (first artifact)
  K=K majority                 : the standard pool majority
  K=K union-any (oracle)       : ceiling — correct under ≥1 of K reads

Sidecar mechanism analysis:
  - identify samples where union-any is correct but plain majority is wrong
  - dump their per-K answers, top_scores, correctness, and base answer
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import analyze_reliability as e1  # noqa: E402
import analyze_reliability_retry as e4  # noqa: E402
import evaluate_true_retry as e_eval  # noqa: E402


DEFAULT_OUT_DIR = e4.DEFAULT_OUT_DIR


def _load_retry_with_top_scores(path):
    """Load a retry artifact, keyed by question, with vote_summary + answer info."""
    with open(path) as f:
        data = json.load(f)
    out = {}
    for g in data["generations"]:
        vs = g.get("vote_summary") or {}
        top_scores = vs.get("top_scores") or []
        margin = None
        top1 = None
        vote_share = None
        if top_scores:
            top1 = float(top_scores[0]["score"])
            total = sum(float(s["score"]) for s in top_scores)
            if total > 0:
                vote_share = top1 / total
            if len(top_scores) >= 2:
                margin = top1 - float(top_scores[1]["score"])
            else:
                margin = top1  # single-candidate sample: margin defined as top1
        out[g["question"]] = {
            "question": g["question"],
            "ground_truth": g["ground_truth"],
            "vote_answer": g.get("vote_answer"),
            "top_scores": top_scores,
            "top1": top1,
            "margin": margin,
            "vote_share": vote_share,
            "valid_events": vs.get("valid_events", 0),
        }
    return out


def _flagged_test_rows(selective_manifest_path, base_artifact, prob_artifact, block_artifact):
    """Return GSM8K test rows restricted to the flagged subset (via selective manifest)."""
    test_rows = e_eval._score_rows(base_artifact, prob_artifact, block_artifact, "test")
    with open(selective_manifest_path) as f:
        man = json.load(f)
    wanted = set(int(i) for i in man["sample_indices"])
    return [r for r in test_rows if r["sample_index"] in wanted], test_rows


def _aggregate(per_sample_K_reads, base_answer, gt):
    """Given K dicts per sample (each with answer/margin/top1/vote_share), apply each aggregator.

    Returns dict { aggregator_name: chosen_answer }.
    """
    # Strip None entries: a retry might have valid_events=0 and thus no answer.
    reads = [r for r in per_sample_K_reads if r and r.get("vote_answer") is not None]

    # K=1 reference: first retry artifact's answer (whichever order)
    k1_answer = per_sample_K_reads[0]["vote_answer"] if per_sample_K_reads else None

    # majority (standard, first-occurrence tie-break)
    counts = Counter([r["vote_answer"] for r in reads])
    if not counts:
        majority = None
    else:
        max_count = max(counts.values())
        tied = [a for a in counts if counts[a] == max_count]
        # first-occurrence-among-tied
        for r in reads:
            if r["vote_answer"] in tied:
                majority = r["vote_answer"]
                break

    # union-any: correct iff any of K answers equals gt
    union_any_correct = int(any(e1._is_correct(r["vote_answer"], gt) for r in reads))

    # best_of_K_by_margin
    if reads and any(r["margin"] is not None for r in reads):
        bestM = max((r for r in reads if r["margin"] is not None), key=lambda r: r["margin"])
        best_by_margin = bestM["vote_answer"]
    else:
        best_by_margin = None

    # best_of_K_by_top1
    if reads and any(r["top1"] is not None for r in reads):
        bestT = max((r for r in reads if r["top1"] is not None), key=lambda r: r["top1"])
        best_by_top1 = bestT["vote_answer"]
    else:
        best_by_top1 = None

    # confidence_weighted_vote (sum top1 by answer across reads)
    weighted = {}
    for r in reads:
        if r["vote_answer"] is None or r["top1"] is None:
            continue
        weighted[r["vote_answer"]] = weighted.get(r["vote_answer"], 0.0) + r["top1"]
    confidence_weighted = max(weighted.items(), key=lambda kv: kv[1])[0] if weighted else None

    # 2_of_K_else_base
    if counts:
        max_count = max(counts.values())
        if max_count >= 2:
            tied = [a for a in counts if counts[a] == max_count]
            for r in reads:
                if r["vote_answer"] in tied:
                    two_of_K = r["vote_answer"]
                    break
        else:
            two_of_K = base_answer
    else:
        two_of_K = base_answer

    return {
        "k1": k1_answer,
        "majority": majority,
        "union_any_correct": union_any_correct,
        "best_by_margin": best_by_margin,
        "best_by_top1": best_by_top1,
        "confidence_weighted": confidence_weighted,
        "two_of_K_else_base": two_of_K,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-artifact", default=e4.DEFAULT_GSM8K_BASE)
    parser.add_argument("--prob-artifact", default=e4.DEFAULT_GSM8K_PROB)
    parser.add_argument("--block-artifact", default=e4.DEFAULT_GSM8K_BLOCKACTIVE)
    parser.add_argument("--selective-manifest", required=True,
                        help="GSM8K selective manifest json (flagged 60 sample indices)")
    parser.add_argument("--retry-artifact", action="append", required=True,
                        help="K=1, K=2, K=3 retry artifacts in order")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--output-stem", default="")
    args = parser.parse_args()

    K = len(args.retry_artifact)
    print(f"[load] K={K} retry artifacts")

    flagged_rows, full_test_rows = _flagged_test_rows(
        args.selective_manifest, args.base_artifact, args.prob_artifact, args.block_artifact)
    print(f"  flagged on test: {len(flagged_rows)}  full test: {len(full_test_rows)}")

    retries = [_load_retry_with_top_scores(p) for p in args.retry_artifact]

    # === Per-flagged-sample aggregator outputs ===
    per_sample = []
    for r in flagged_rows:
        q = r["question"]
        gt = r["ground_truth"]
        base = r["exp_only_answer"]
        reads = [retries[k].get(q) for k in range(K)]
        agg = _aggregate(reads, base, gt)
        per_sample.append({
            "sample_index": r["sample_index"],
            "ground_truth": gt,
            "base_answer": base,
            "base_correct": int(e1._is_correct(base, gt)),
            "K_reads": [
                {
                    "answer": reads[k]["vote_answer"] if reads[k] else None,
                    "correct": int(e1._is_correct(reads[k]["vote_answer"] if reads[k] else None, gt)),
                    "margin": reads[k]["margin"] if reads[k] else None,
                    "top1": reads[k]["top1"] if reads[k] else None,
                    "valid_events": reads[k]["valid_events"] if reads[k] else 0,
                }
                for k in range(K)
            ],
            "aggregates": agg,
            "aggregate_correct": {
                name: int(e1._is_correct(ans, gt)) if name != "union_any_correct" else agg[name]
                for name, ans in agg.items()
            },
        })

    # === Aggregate accuracy on flagged subset ===
    n = len(per_sample)
    def acc(key):
        return sum(s["aggregate_correct"][key] for s in per_sample) / n

    aggregator_names = ["k1", "majority", "best_by_margin", "best_by_top1",
                        "confidence_weighted", "two_of_K_else_base", "union_any_correct"]
    base_acc_flagged = sum(s["base_correct"] for s in per_sample) / n

    print(f"\n[flagged subset accuracy, n={n}]")
    print(f"  base (no retry):       {base_acc_flagged*100:.2f}%  ({int(base_acc_flagged*n)}/{n})")
    for name in aggregator_names:
        a = acc(name)
        print(f"  {name:25s}  {a*100:.2f}%  ({int(a*n)}/{n})")

    # === Full-test deploy with each aggregator on flagged, base elsewhere ===
    full_n = len(full_test_rows)
    full_base_correct = sum(int(e1._is_correct(r["exp_only_answer"], r["ground_truth"])) for r in full_test_rows)
    full_base_acc = full_base_correct / full_n
    print(f"\n[full-test deploy, n={full_n}; base_acc={full_base_acc*100:.2f}%]")

    flagged_index = {s["sample_index"]: s for s in per_sample}
    full_test_acc = {}
    full_test_fixes_hurts = {}
    for name in aggregator_names:
        if name == "union_any_correct":
            continue
        correct = 0
        fixes = 0
        hurts = 0
        for r in full_test_rows:
            gt = r["ground_truth"]
            if r["sample_index"] in flagged_index:
                ans = flagged_index[r["sample_index"]]["aggregates"][name]
                if ans is None:
                    ans = r["exp_only_answer"]
            else:
                ans = r["exp_only_answer"]
            is_correct = int(e1._is_correct(ans, gt))
            correct += is_correct
            if r["sample_index"] in flagged_index:
                base_corr = flagged_index[r["sample_index"]]["base_correct"]
                if is_correct and not base_corr:
                    fixes += 1
                if (not is_correct) and base_corr:
                    hurts += 1
        full_test_acc[name] = correct / full_n
        full_test_fixes_hurts[name] = (fixes, hurts)
        delta = (correct / full_n) - full_base_acc
        print(f"  {name:25s}  {correct/full_n*100:.2f}%  fixes={fixes} hurts={hurts}  delta {delta*100:+.2f}pt")

    # === Mechanism sidecar: where does union-any rescue but majority miss? ===
    union_rescue_majority_miss = [
        s for s in per_sample
        if s["aggregate_correct"]["union_any_correct"] == 1 and s["aggregate_correct"]["majority"] == 0
    ]
    print(f"\n[mechanism] samples where union-any correct but majority miss: {len(union_rescue_majority_miss)}")
    print(f"  (these are the 9-or-so samples explaining the union-vs-majority gap)")

    # === Output ===
    date_str = datetime.now().strftime("%Y%m%d")
    stem = args.output_stem or f"retry_pool_aggregators_K{K}_{date_str}"
    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, f"{stem}.json")
    md_path = os.path.join(args.out_dir, f"{stem}.md")

    out = {
        "config": {"K": K, "retry_artifacts": list(args.retry_artifact),
                   "selective_manifest": args.selective_manifest},
        "flagged_n": n,
        "full_test_n": full_n,
        "base_acc_flagged": base_acc_flagged,
        "full_base_acc": full_base_acc,
        "flagged_aggregator_acc": {name: acc(name) for name in aggregator_names},
        "full_test_aggregator_acc": full_test_acc,
        "full_test_fixes_hurts": {k: list(v) for k, v in full_test_fixes_hurts.items()},
        "per_sample": per_sample,
        "mechanism_union_rescue_majority_miss_indices": [s["sample_index"] for s in union_rescue_majority_miss],
    }
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote: {json_path}")

    # === Markdown ===
    lines = []
    lines.append(f"# Retry-Pool Alternative Aggregators — K={K} — {date_str}")
    lines.append("")
    lines.append(f"- retry artifacts: {[os.path.basename(os.path.dirname(p)) for p in args.retry_artifact]}")
    lines.append(f"- selective manifest: `{args.selective_manifest}`")
    lines.append(f"- flagged subset on test: `{n}`; full test: `{full_n}`")
    lines.append("")
    lines.append("## Flagged-subset accuracy by aggregator")
    lines.append("")
    lines.append("| Aggregator | Correct / flagged | Acc |")
    lines.append("|---|---:|---:|")
    lines.append(f"| **base (no retry)** | {int(base_acc_flagged*n)} / {n} | **{base_acc_flagged*100:.2f}%** |")
    for name in aggregator_names:
        a = acc(name)
        marker = " ←" if name == "union_any_correct" else ""
        lines.append(f"| {name}{marker} | {int(a*n)} / {n} | {a*100:.2f}% |")
    lines.append("")
    lines.append("`union_any_correct` is the oracle ceiling; the others are deployable.")
    lines.append("")
    lines.append("## Full-test deploy (n=264, aggregator on flagged, base elsewhere)")
    lines.append("")
    lines.append("| Aggregator | Full-test acc | delta vs base | fixes | hurts |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(f"| **P0 baseline (no retry)** | {full_base_acc*100:.2f}% | — | 0 | 0 |")
    for name in aggregator_names:
        if name == "union_any_correct":
            continue
        a = full_test_acc[name]
        fx, ht = full_test_fixes_hurts[name]
        delta = a - full_base_acc
        lines.append(f"| {name} | {a*100:.2f}% | `{delta*100:+.2f}pt` | {fx} | {ht} |")
    lines.append("")
    lines.append("## Mechanism sidecar — samples where union-any correct but majority miss")
    lines.append("")
    lines.append(f"`{len(union_rescue_majority_miss)}` samples on flagged 60. These are the source of the oracle-vs-majority gap.")
    lines.append("")
    lines.append("| idx | gt | base | K1 ans (corr) | K2 ans (corr) | K3 ans (corr) | majority | best_margin | best_top1 | conf_weighted | 2-of-K-else-base | margins |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for s in union_rescue_majority_miss:
        margins = [f"{r['margin']:.1f}" if r['margin'] is not None else "—" for r in s["K_reads"]]
        agg = s["aggregates"]
        agg_corr = s["aggregate_correct"]
        def mark(name):
            return f"{agg[name]} ({'✓' if agg_corr[name] else '✗'})"
        kreads_str = [f"{r['answer']} ({'✓' if r['correct'] else '✗'})" for r in s["K_reads"]]
        lines.append(
            f"| {s['sample_index']} | {s['ground_truth']} | {s['base_answer']} "
            f"| {kreads_str[0]} | {kreads_str[1] if K>=2 else '—'} | {kreads_str[2] if K>=3 else '—'} "
            f"| {mark('majority')} | {mark('best_by_margin')} | {mark('best_by_top1')} "
            f"| {mark('confidence_weighted')} | {mark('two_of_K_else_base')} | {' / '.join(margins)} |"
        )
    lines.append("")
    lines.append("Reading guide:")
    lines.append("")
    lines.append("- if `best_by_margin` is correct on most of these `(✓)` while `majority` is wrong, then confidence-margin can pull the oracle gain out — best-of-K by confidence works.")
    lines.append("- if `2_of_K_else_base` is correct on most because base was correct or because some agreement materializes, the safer policy beats majority by avoiding the wrong 2-of-3 cluster.")
    lines.append("- if both alternatives still miss most of these, the diversity is real but the in-vote confidence signal doesn't disambiguate which K-vote to trust.")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
