"""Evaluate a true selective-retry artifact against the offline fallback policy.

Policy family:
  - P0: keep base `exp_only`
  - P1: offline fallback (`prob_vote` on low-score samples)
  - P2: true retry, use retry artifact `exp_only_answer` on low-score samples
  - P3: true retry, use retry artifact selected answer on low-score samples

The retry artifact may be a subset rerun. In that case only questions present in
the retry artifact are eligible for replacement; the rest stay on the base path.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import analyze_reliability_retry as e4


DEFAULT_OUT_DIR = e4.DEFAULT_OUT_DIR
DEFAULT_SCORE_KEY = "logistic_broad_score"
DEFAULT_TAU = 0.5845


def _score_rows(base_artifact, prob_artifact=None, block_artifact=None, eval_split="test"):
    feat_rows = e4._load_feature_rows(base_artifact)
    base_answers = e4._load_answer_artifact(base_artifact)
    prob_answers = e4._load_answer_artifact(prob_artifact) if prob_artifact else base_answers
    block_answers = e4._load_answer_artifact(block_artifact) if block_artifact else base_answers
    merged = e4._merge_sources(feat_rows, base_answers, prob_answers, block_answers)
    split = e4._fit_scores(merged)
    if eval_split == "val":
        rows = split["val"]
    elif eval_split == "test":
        rows = split["test"]
    elif eval_split == "full":
        rows = split["val"] + split["test"]
    else:
        raise ValueError(f"Unsupported eval_split: {eval_split}")
    rows = sorted(rows, key=lambda r: r["sample_index"])
    return rows


def _load_retry_answers(path):
    rows = e4._load_answer_artifact(path)
    return {r["question"]: r for r in rows}


def _evaluate_policy(rows, score_key, tau, retry_map, retry_answer_kind):
    baseline_acc = e4._accuracy(rows, "exp_only_answer")
    offline_prob_acc = 0
    retry_exp_acc = 0
    retry_policy_acc = 0

    offline_changed = retry_exp_changed = retry_policy_changed = 0
    offline_fix = retry_exp_fix = retry_policy_fix = 0
    offline_hurt = retry_exp_hurt = retry_policy_hurt = 0
    retry_eligible = 0
    retry_missing = 0

    def _check_delta(base, new, gt):
        fix = int(e4.e1._is_correct(new, gt) and not e4.e1._is_correct(base, gt))
        hurt = int((not e4.e1._is_correct(new, gt)) and e4.e1._is_correct(base, gt))
        return fix, hurt

    for r in rows:
        gt = r["ground_truth"]
        base = r["exp_only_answer"]
        flagged = float(r[score_key]) <= tau
        retry_row = retry_map.get(r["question"])

        offline_prob = r.get("prob_vote") if flagged else base
        if offline_prob != base:
            offline_changed += 1
            fix, hurt = _check_delta(base, offline_prob, gt)
            offline_fix += fix
            offline_hurt += hurt
        offline_prob_acc += int(e4.e1._is_correct(offline_prob, gt))

        retry_exp = base
        retry_policy = base
        if flagged:
            retry_eligible += 1
            if retry_row is None:
                retry_missing += 1
            else:
                retry_exp = retry_row["exp_only_answer"]
                if retry_answer_kind == "vote":
                    retry_policy = retry_row["stored_vote_answer"]
                elif retry_answer_kind == "final":
                    retry_policy = retry_row["stored_final_answer"]
                elif retry_answer_kind == "exp_only":
                    retry_policy = retry_row["exp_only_answer"]
                else:
                    raise ValueError(f"Unsupported retry_answer_kind: {retry_answer_kind}")
                if retry_policy is None:
                    retry_policy = base

        if retry_exp != base:
            retry_exp_changed += 1
            fix, hurt = _check_delta(base, retry_exp, gt)
            retry_exp_fix += fix
            retry_exp_hurt += hurt
        if retry_policy != base:
            retry_policy_changed += 1
            fix, hurt = _check_delta(base, retry_policy, gt)
            retry_policy_fix += fix
            retry_policy_hurt += hurt

        retry_exp_acc += int(e4.e1._is_correct(retry_exp, gt))
        retry_policy_acc += int(e4.e1._is_correct(retry_policy, gt))

    n = len(rows)
    return {
        "score_key": score_key,
        "tau": float(tau),
        "retry_answer_kind": retry_answer_kind,
        "n_total": n,
        "retry_eligible": retry_eligible,
        "retry_rate": retry_eligible / n if n else 0.0,
        "retry_missing": retry_missing,
        "retry_coverage": (retry_eligible - retry_missing) / retry_eligible if retry_eligible else 0.0,
        "baseline": {
            "acc": baseline_acc,
        },
        "offline_prob_fallback": {
            "acc": offline_prob_acc / n if n else 0.0,
            "delta_vs_base": (offline_prob_acc / n - baseline_acc) if n else 0.0,
            "changed": offline_changed,
            "fixes": offline_fix,
            "hurts": offline_hurt,
        },
        "retry_exp_only": {
            "acc": retry_exp_acc / n if n else 0.0,
            "delta_vs_base": (retry_exp_acc / n - baseline_acc) if n else 0.0,
            "changed": retry_exp_changed,
            "fixes": retry_exp_fix,
            "hurts": retry_exp_hurt,
        },
        "retry_policy": {
            "acc": retry_policy_acc / n if n else 0.0,
            "delta_vs_base": (retry_policy_acc / n - baseline_acc) if n else 0.0,
            "changed": retry_policy_changed,
            "fixes": retry_policy_fix,
            "hurts": retry_policy_hurt,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-artifact", default=e4.DEFAULT_GSM8K_BASE)
    parser.add_argument("--prob-artifact", default=e4.DEFAULT_GSM8K_PROB)
    parser.add_argument("--block-artifact", default=e4.DEFAULT_GSM8K_BLOCKACTIVE)
    parser.add_argument("--retry-artifact", required=True)
    parser.add_argument("--score-key", choices=["combined_score", "logistic_broad_score"], default=DEFAULT_SCORE_KEY)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--eval-split", choices=["val", "test", "full"], default="test")
    parser.add_argument("--retry-answer-kind", choices=["vote", "final", "exp_only"], default="vote")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--output-stem", default="")
    args = parser.parse_args()

    rows = _score_rows(args.base_artifact, args.prob_artifact, args.block_artifact, args.eval_split)
    retry_map = _load_retry_answers(args.retry_artifact)
    result = _evaluate_policy(rows, args.score_key, args.tau, retry_map, args.retry_answer_kind)
    result["base_artifact"] = args.base_artifact
    result["retry_artifact"] = args.retry_artifact
    result["eval_split"] = args.eval_split

    date_str = datetime.now().strftime("%Y%m%d")
    stem = args.output_stem or f"true_retry_eval_{args.score_key}_{date_str}"
    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, f"{stem}.json")
    md_path = os.path.join(args.out_dir, f"{stem}.md")

    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    lines = []
    lines.append(f"# True Retry Evaluation — {date_str}")
    lines.append("")
    lines.append(f"- base artifact: `{args.base_artifact}`")
    lines.append(f"- retry artifact: `{args.retry_artifact}`")
    lines.append(f"- score key: `{args.score_key}`")
    lines.append(f"- tau: `{args.tau:.4f}`")
    lines.append(f"- eval split: `{args.eval_split}`")
    lines.append(f"- retry answer kind: `{args.retry_answer_kind}`")
    lines.append("")
    lines.append("## Retry coverage")
    lines.append("")
    lines.append(f"- retry eligible: `{result['retry_eligible']}/{result['n_total']}` (`{result['retry_rate']:.2%}`)")
    lines.append(f"- retry answers available: `{result['retry_eligible'] - result['retry_missing']}`")
    lines.append(f"- retry coverage among eligible: `{result['retry_coverage']:.2%}`")
    lines.append("")
    lines.append("## Policies")
    lines.append("")
    lines.append(f"- baseline `exp_only`: `{result['baseline']['acc']:.2%}`")
    p1 = result["offline_prob_fallback"]
    lines.append(
        f"- P1 offline prob fallback: `{p1['acc']:.2%}` (`{p1['delta_vs_base']:+.2%}`), "
        f"changed `{p1['changed']}`, fixes `{p1['fixes']}`, hurts `{p1['hurts']}`"
    )
    p2 = result["retry_exp_only"]
    lines.append(
        f"- P2 true retry (retry exp_only): `{p2['acc']:.2%}` (`{p2['delta_vs_base']:+.2%}`), "
        f"changed `{p2['changed']}`, fixes `{p2['fixes']}`, hurts `{p2['hurts']}`"
    )
    p3 = result["retry_policy"]
    lines.append(
        f"- P3 true retry ({args.retry_answer_kind}): `{p3['acc']:.2%}` (`{p3['delta_vs_base']:+.2%}`), "
        f"changed `{p3['changed']}`, fixes `{p3['fixes']}`, hurts `{p3['hurts']}`"
    )

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
