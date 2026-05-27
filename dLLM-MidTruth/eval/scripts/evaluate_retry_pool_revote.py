"""Phase E-Retry-Pool: step-level pooled re-vote across K retry artifacts.

This complements `evaluate_retry_pool.py`, which aggregates one answer per run
via majority. Here we instead pool the valid vote-debug events from the first K
retry artifacts and re-run the original `exp_only` voting rule over the merged
event set.

Question:
  If regeneration diversity exists, does *step-level evidence pooling* recover
  more value than simple majority over per-run answers?
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import scripts.analyze_reliability as e1  # noqa: E402
import scripts.analyze_reliability_retry as e4  # noqa: E402
import scripts.evaluate_true_retry as e_eval  # noqa: E402


DEFAULT_OUT_DIR = e4.DEFAULT_OUT_DIR


def _load_events_by_question(path):
    with open(path) as f:
        data = json.load(f)

    out = {}
    for gen, vd in zip(data["generations"], data["vote_debug"]):
        valid = []
        for step in vd["steps"]:
            if step.get("skip_reason") is not None:
                continue
            pa = step.get("parsed_answer")
            if pa is None:
                continue
            valid.append(
                {
                    "step": step["step"],
                    "total_steps": step["total_steps"],
                    "parsed_answer": pa,
                    "exp_weight": math.exp(step["step"] / step["total_steps"] * e1.ALPHA),
                }
            )
        out[gen["question"]] = {
            "question": gen["question"],
            "ground_truth": gen["ground_truth"],
            "valid_events": valid,
        }
    return out


def _pooled_exp_vote(valid_events):
    if not valid_events:
        return None
    return e1._exp_only_vote(valid_events)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-artifact", default=e4.DEFAULT_GSM8K_BASE)
    parser.add_argument("--prob-artifact", default=e4.DEFAULT_GSM8K_PROB)
    parser.add_argument("--block-artifact", default=e4.DEFAULT_GSM8K_BLOCKACTIVE)
    parser.add_argument("--score-key", default="logistic_broad_score")
    parser.add_argument("--tau", type=float, default=0.5845)
    parser.add_argument("--retry-artifact", action="append", required=True)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--output-stem", default="")
    args = parser.parse_args()

    rows = e_eval._score_rows(args.base_artifact, args.prob_artifact, args.block_artifact, "test")
    flagged = [r for r in rows if float(r[args.score_key]) <= args.tau]
    retry_maps = [_load_events_by_question(path) for path in args.retry_artifact]
    K = len(retry_maps)

    print(f"[load] K = {K} retry artifacts")
    print(f"  flagged on test: {len(flagged)}/{len(rows)}")

    per_sample = []
    for r in flagged:
        gt = r["ground_truth"]
        q = r["question"]
        pooled_votes = []
        pooled_event_counts = []
        merged = []
        for rm in retry_maps:
            events = rm.get(q, {}).get("valid_events", [])
            merged.extend(events)
            pooled_event_counts.append(len(merged))
            pooled_votes.append(_pooled_exp_vote(merged))

        per_sample.append(
            {
                "sample_index": r["sample_index"],
                "question": q,
                "ground_truth": gt,
                "base_correct": int(e1._is_correct(r["exp_only_answer"], gt)),
                "pooled_votes": pooled_votes,
                "pooled_correct_at_k": [int(e1._is_correct(v, gt)) for v in pooled_votes],
                "pooled_event_counts": pooled_event_counts,
            }
        )

    pooled_acc_at_k = []
    pooled_new_rescue_at_k = []
    seen_rescued = set()
    for k in range(K):
        pooled_acc_at_k.append(sum(s["pooled_correct_at_k"][k] for s in per_sample) / len(per_sample))
        new_rescue = 0
        for s in per_sample:
            if s["base_correct"]:
                continue
            if s["pooled_correct_at_k"][k] and s["sample_index"] not in seen_rescued:
                new_rescue += 1
                seen_rescued.add(s["sample_index"])
        pooled_new_rescue_at_k.append(new_rescue)

    print("\n[per-K pooled exp re-vote on flagged subset]")
    print(f"{'k':>3}  {'pooled_acc':>11}  {'new_rescue':>11}  {'cum_rescued':>12}")
    cum = 0
    for k in range(K):
        cum += pooled_new_rescue_at_k[k]
        print(f"{k+1:>3}  {pooled_acc_at_k[k]*100:>10.2f}%  {pooled_new_rescue_at_k[k]:>11d}  {cum:>12d}")

    full_acc_at_k = []
    for k in range(K):
        correct = 0
        for r in rows:
            gt = r["ground_truth"]
            if float(r[args.score_key]) <= args.tau:
                sample = next(s for s in per_sample if s["question"] == r["question"])
                use = sample["pooled_votes"][k]
                if use is None:
                    use = r["exp_only_answer"]
            else:
                use = r["exp_only_answer"]
            correct += int(e1._is_correct(use, gt))
        full_acc_at_k.append(correct / len(rows))

    print("\n[full-test deploy: pooled exp re-vote(K) on flagged, base elsewhere]")
    for k in range(K):
        print(f"  K={k+1}: {full_acc_at_k[k]*100:.2f}%")

    os.makedirs(args.out_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    stem = args.output_stem or f"retry_pool_revote_eval_K{K}_{date_str}"

    out = {
        "config": {
            "K": K,
            "score_key": args.score_key,
            "tau": args.tau,
            "retry_artifacts": list(args.retry_artifact),
        },
        "n_flagged": len(flagged),
        "per_k_pooled_exp_acc": pooled_acc_at_k,
        "per_k_pooled_exp_new_rescue": pooled_new_rescue_at_k,
        "per_k_full_test_pooled_exp_acc": full_acc_at_k,
        "per_sample": per_sample,
    }
    json_path = os.path.join(args.out_dir, f"{stem}.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote: {json_path}")

    lines = []
    lines.append(f"# Retry Pool Re-vote Evaluation — K={K} — {date_str}")
    lines.append("")
    lines.append("- Aggregator: pool valid step events from the first `K` retry artifacts and re-run the original `exp_only` vote over the merged event set.")
    lines.append(f"- flagged subset on test: `{len(flagged)}` samples")
    lines.append("")
    lines.append("## Per-K pooled exp re-vote on flagged subset")
    lines.append("")
    lines.append("| K | pooled exp acc | new rescues at K | cumulative rescues |")
    lines.append("|---:|---:|---:|---:|")
    cum = 0
    for k in range(K):
        cum += pooled_new_rescue_at_k[k]
        lines.append(f"| {k+1} | {pooled_acc_at_k[k]*100:.2f}% | {pooled_new_rescue_at_k[k]} | {cum} |")
    lines.append("")
    lines.append("## Full-test deploy: pooled exp re-vote(K) on flagged, base elsewhere")
    lines.append("")
    lines.append("| K | full-test acc |")
    lines.append("|---:|---:|")
    for k in range(K):
        lines.append(f"| {k+1} | {full_acc_at_k[k]*100:.2f}% |")
    md_path = os.path.join(args.out_dir, f"{stem}.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
