"""Phase E-Retry-Pool evaluator.

Consumes K retry artifacts on the same flagged sample set and aggregates
them as a regeneration-diversity pool. The K artifacts should be fresh
regeneration runs (temperature > 0 or different seeds), not vote-method
variants on the same trajectory.

Reports, as K grows:

  - per-K union-any (correct under at least one of the first K reads): upper
    bound under an oracle picker
  - per-K majority vote (most common answer among K reads): a deployable
    aggregation; ties resolved by first-appearance order
  - per-K uniquely-rescued count (samples that none of the first K-1 reads
    rescued, but the K-th does — measures marginal contribution of each new run)
  - full-test policy accuracy under each aggregator, when used to replace
    `exp_only` on the flagged subset

Usage
  evaluate_retry_pool.py \
    --score-key logistic_broad_score --tau 0.5845 \
    --retry-artifact /path/to/retry1/rank_0_generations.json \
    --retry-artifact /path/to/retry2/rank_0_generations.json \
    --retry-artifact /path/to/retry3/rank_0_generations.json \
    --answer-kind exp_only
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

import scripts.analyze_reliability as e1  # noqa: E402
import scripts.analyze_reliability_retry as e4  # noqa: E402
import scripts.evaluate_true_retry as e_eval  # noqa: E402


DEFAULT_OUT_DIR = e4.DEFAULT_OUT_DIR


def _read_answer(retry_row, answer_kind):
    if retry_row is None:
        return None
    if answer_kind == "exp_only":
        return retry_row.get("exp_only_answer")
    if answer_kind == "vote":
        return retry_row.get("stored_vote_answer")
    if answer_kind == "final":
        return retry_row.get("stored_final_answer")
    raise ValueError(answer_kind)


def _majority(answers):
    """Majority vote over a list of answers (None treated as 'no vote' and skipped).
    Ties broken by first-appearance order. Returns None if no non-None answers."""
    cleaned = [a for a in answers if a is not None]
    if not cleaned:
        return None
    counts = Counter(cleaned)
    max_count = max(counts.values())
    tied = [a for a in cleaned if counts[a] == max_count]
    # first-occurrence-among-tied
    seen = set()
    for a in cleaned:
        if a in tied and a not in seen:
            return a
        seen.add(a)
    return cleaned[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-artifact", default=e4.DEFAULT_GSM8K_BASE)
    parser.add_argument("--prob-artifact", default=e4.DEFAULT_GSM8K_PROB)
    parser.add_argument("--block-artifact", default=e4.DEFAULT_GSM8K_BLOCKACTIVE)
    parser.add_argument("--score-key", default="logistic_broad_score")
    parser.add_argument("--tau", type=float, default=0.5845)
    parser.add_argument("--retry-artifact", action="append", required=True,
                        help="path to a retry artifact's rank_0_generations.json; pass multiple times")
    parser.add_argument("--answer-kind", choices=["exp_only", "vote", "final"], default="exp_only")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--output-stem", default="")
    args = parser.parse_args()

    K = len(args.retry_artifact)
    print(f"[load] K = {K} retry artifacts")
    rows = e_eval._score_rows(args.base_artifact, args.prob_artifact, args.block_artifact, "test")
    flagged = [r for r in rows if float(r[args.score_key]) <= args.tau]
    print(f"  flagged on test: {len(flagged)}/{len(rows)}")

    # Load each retry artifact as a question -> retry_row dict
    retry_maps = []
    sanity_answer_equal = []  # compare deployed answers against the first artifact
    ref_artifact = args.retry_artifact[0]

    for path in args.retry_artifact:
        retry_maps.append({r["question"]: r for r in e4._load_answer_artifact(path)})

    ref_map = retry_maps[0]
    for path, rm in zip(args.retry_artifact, retry_maps):
        eq = 0
        diff = 0
        for q, ref_row in ref_map.items():
            if q not in rm:
                continue
            ref_ans = _read_answer(ref_row, args.answer_kind)
            cur_ans = _read_answer(rm[q], args.answer_kind)
            if ref_ans == cur_ans:
                eq += 1
            else:
                diff += 1
        sanity_answer_equal.append({"path": path, "answer_equal_with_ref": eq, "answer_diff_with_ref": diff})

    print(f"\n[answer diversity sanity]")
    for s in sanity_answer_equal:
        print(f"  ref vs {os.path.basename(os.path.dirname(s['path']))}: answer equal {s['answer_equal_with_ref']}, diff {s['answer_diff_with_ref']}")
    print(f"  (diff > 0 means the additional artifact contributes at least some different answers under the selected answer_kind)")

    # Per-flagged-sample reads across K
    per_sample = []
    for r in flagged:
        gt = r["ground_truth"]
        reads = []
        for rm in retry_maps:
            reads.append(_read_answer(rm.get(r["question"]), args.answer_kind))
        correct_per_k = [int(e1._is_correct(a, gt)) for a in reads]
        # Cumulative union and majority as K grows
        union_at_k = []
        majority_at_k = []
        cur_union = 0
        for k in range(1, K + 1):
            cur_union = 1 if any(correct_per_k[:k]) else 0
            union_at_k.append(cur_union)
            maj = _majority(reads[:k])
            majority_at_k.append(int(e1._is_correct(maj, gt)))
        per_sample.append({
            "sample_index": r["sample_index"],
            "base_correct": int(e1._is_correct(r["exp_only_answer"], gt)),
            "reads": reads,
            "correct_per_k": correct_per_k,
            "union_at_k": union_at_k,
            "majority_at_k": majority_at_k,
        })

    n_flagged = len(per_sample)
    # Aggregate per K
    union_acc_at_k = []
    majority_acc_at_k = []
    uniquely_rescued_at_k = []  # marginal: new fixes added by k-th read
    seen_rescued = set()
    for k in range(K):
        ua = sum(s["union_at_k"][k] for s in per_sample) / n_flagged
        ma = sum(s["majority_at_k"][k] for s in per_sample) / n_flagged
        union_acc_at_k.append(ua)
        majority_acc_at_k.append(ma)
        new_rescue_count = 0
        for s in per_sample:
            if s["base_correct"]:
                continue
            if s["correct_per_k"][k] and s["sample_index"] not in seen_rescued:
                new_rescue_count += 1
                seen_rescued.add(s["sample_index"])
        uniquely_rescued_at_k.append(new_rescue_count)

    print(f"\n[per-K aggregate on flagged {n_flagged}]")
    print(f"{'k':>3}  {'union_acc':>10}  {'majority_acc':>13}  {'new_rescue':>11}  {'cum_rescued':>12}")
    cum = 0
    for k in range(K):
        cum += uniquely_rescued_at_k[k]
        print(f"{k+1:>3}  {union_acc_at_k[k]*100:>9.2f}%  {majority_acc_at_k[k]*100:>12.2f}%  {uniquely_rescued_at_k[k]:>11d}  {cum:>12d}")

    # Full-test accuracies if we deploy majority on flagged samples
    full_acc_at_k = []
    for k in range(K):
        correct = 0
        for r in rows:
            gt = r["ground_truth"]
            if float(r[args.score_key]) <= args.tau:
                reads = [_read_answer(rm.get(r["question"]), args.answer_kind) for rm in retry_maps[:k+1]]
                use = _majority(reads)
                if use is None:
                    use = r["exp_only_answer"]
            else:
                use = r["exp_only_answer"]
            correct += int(e1._is_correct(use, gt))
        full_acc_at_k.append(correct / len(rows))

    print(f"\n[full-test deploy: majority(K) on flagged, base elsewhere]")
    for k in range(K):
        print(f"  K={k+1}: {full_acc_at_k[k]*100:.2f}%")

    # === Write ===
    os.makedirs(args.out_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    stem = args.output_stem or f"retry_pool_eval_K{K}_{date_str}"

    out = {
        "config": {
            "K": K, "score_key": args.score_key, "tau": args.tau,
            "answer_kind": args.answer_kind,
            "retry_artifacts": list(args.retry_artifact),
        },
        "answer_diversity_sanity": sanity_answer_equal,
        "n_flagged": n_flagged,
        "per_k_union_acc": union_acc_at_k,
        "per_k_majority_acc": majority_acc_at_k,
        "per_k_new_rescue": uniquely_rescued_at_k,
        "per_k_full_test_majority_acc": full_acc_at_k,
        "per_sample": per_sample,
    }
    json_path = os.path.join(args.out_dir, f"{stem}.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote: {json_path}")

    lines = []
    lines.append(f"# Phase E-Retry-Pool — K={K} — {date_str}")
    lines.append("")
    lines.append(f"- K = `{K}` retry artifacts, answer_kind=`{args.answer_kind}`")
    lines.append(f"- flagged subset on test: `{n_flagged}` samples")
    lines.append("")
    lines.append("## Answer diversity sanity (vs first artifact)")
    lines.append("")
    lines.append("| artifact | answer equal with ref | answer diff with ref |")
    lines.append("|---|---:|---:|")
    for s in sanity_answer_equal:
        lines.append(f"| `{os.path.basename(os.path.dirname(s['path']))}` | {s['answer_equal_with_ref']} | {s['answer_diff_with_ref']} |")
    lines.append("")
    lines.append("This sanity block is answer-level, not raw-text-level: these retry artifacts do not preserve enough generation text to safely compare trajectories directly. If answer-diff with ref is `0` for all non-ref artifacts, the additional runs did not contribute any new deployed answers under the selected answer_kind, even if hidden raw trajectories differed.")
    lines.append("")
    lines.append("## Per-K cumulative aggregates on flagged subset")
    lines.append("")
    lines.append("| K | union-any acc | majority acc | new rescues at K | cumulative rescues |")
    lines.append("|---:|---:|---:|---:|---:|")
    cum = 0
    for k in range(K):
        cum += uniquely_rescued_at_k[k]
        lines.append(f"| {k+1} | {union_acc_at_k[k]*100:.2f}% | {majority_acc_at_k[k]*100:.2f}% | {uniquely_rescued_at_k[k]} | {cum} |")
    lines.append("")
    lines.append("- `union-any` is the oracle ceiling at K: correct under at least one of the first K reads.")
    lines.append("- `majority` is a deployable aggregator: majority answer among the first K reads.")
    lines.append("- `new rescues at K` counts samples that the first `K-1` reads did not rescue but the `K`-th did. Marginal contribution.")
    lines.append("")
    lines.append("## Full-test accuracy with majority(K) on flagged, base elsewhere")
    lines.append("")
    lines.append("| K | full-test acc |")
    lines.append("|---:|---:|")
    for k in range(K):
        lines.append(f"| {k+1} | {full_acc_at_k[k]*100:.2f}% |")
    lines.append("")
    lines.append("This is the deployable policy: vote across K independent regenerations on the score-flagged slice, keep `exp_only` on the unflagged. If full-test acc keeps growing with K, additional regeneration seeds keep paying off; if it plateaus, the diversity is saturated.")
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append("- single outer seed (the score-flagged set comes from seed=42 score fit).")
    lines.append("- the answer-diversity sanity rows do **not** prove raw trajectory diversity. They only tell us whether the additional artifacts contribute different deployed answers under the selected answer_kind.")
    md_path = os.path.join(args.out_dir, f"{stem}.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
