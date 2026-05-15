"""Phase E-Retry-Control-MultiSeed evaluator.

Aggregates per-seed random-control retry runs against the same selective
retry. For each seed, computes the targeting marginal
`(P_selective − P_random_seed)` and the subset-local retry delta on the
random subset. Reports per-seed table + mean ± std across seeds.

Use after running multiple random-control retries with different
`--random-seed` values (recommended: `--exclude-flagged` for all so they
stay strict complements). Each random retry artifact pairs with its own
manifest json.

Usage
  evaluate_retry_control_multiseed.py \
    --selective-retry-artifact /path/to/v6/rank_0_generations.json \
    --selective-manifest /path/to/selective_manifest.json \
    --random-pair /path/to/retry1_artifact.json::/path/to/retry1_manifest.json \
    --random-pair /path/to/retry2_artifact.json::/path/to/retry2_manifest.json \
    --random-pair /path/to/retry3_artifact.json::/path/to/retry3_manifest.json \
    --answer-kind exp_only
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
import scripts.evaluate_retry_control as ctrl  # noqa: E402


DEFAULT_OUT_DIR = e4.DEFAULT_OUT_DIR


def _parse_pair(arg):
    if "::" not in arg:
        raise argparse.ArgumentTypeError(f"--random-pair expects ARTIFACT::MANIFEST (got {arg!r})")
    a, m = arg.split("::", 1)
    return a, m


def _mean_std(xs):
    n = len(xs)
    if n == 0:
        return None, None
    m = sum(xs) / n
    if n == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-artifact", default=e4.DEFAULT_GSM8K_BASE)
    parser.add_argument("--prob-artifact", default=e4.DEFAULT_GSM8K_PROB)
    parser.add_argument("--block-artifact", default=e4.DEFAULT_GSM8K_BLOCKACTIVE)
    parser.add_argument("--selective-retry-artifact", required=True)
    parser.add_argument("--selective-manifest", required=True)
    parser.add_argument("--random-pair", required=True, action="append", type=_parse_pair,
                        help="ARTIFACT::MANIFEST for one random-control run; pass multiple times")
    parser.add_argument("--answer-kind", choices=["exp_only", "vote", "final"], default="exp_only")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--output-stem", default="")
    args = parser.parse_args()

    rows = e_eval._score_rows(args.base_artifact, args.prob_artifact, args.block_artifact, "test")
    base_correct = sum(int(e1._is_correct(r["exp_only_answer"], r["ground_truth"])) for r in rows)
    base_acc = base_correct / len(rows)
    print(f"[load] gsm8k test n={len(rows)}, base_acc={base_acc*100:.2f}%")

    selective_qs, sel_manifest = ctrl._load_manifest_question_set(args.selective_manifest, rows)
    sel_map = e_eval._load_retry_answers(args.selective_retry_artifact)
    p_sel = ctrl._decompose_policy(rows, selective_qs, sel_map, args.answer_kind)
    print(f"[selective] acc={p_sel['acc']*100:.2f}%  fixes={len(p_sel['fix_indices'])}  hurts={len(p_sel['hurt_indices'])}")
    print(f"  subset-local: base {p_sel['subset_local']['base_acc']*100:.2f}% → retry {p_sel['subset_local']['retry_acc']*100:.2f}% (delta {p_sel['subset_local']['delta']*100:+.2f}pt)")

    per_seed = []
    for art_path, man_path in args.random_pair:
        random_qs, rand_manifest_json = ctrl._load_manifest_question_set(man_path, rows)
        overlap = len(selective_qs & random_qs)
        rand_map = e_eval._load_retry_answers(art_path)
        p_rand = ctrl._decompose_policy(rows, random_qs, rand_map, args.answer_kind)
        seed = rand_manifest_json.get("random_seed")
        exclude_flagged = rand_manifest_json.get("exclude_flagged", False)
        targeting_marginal = p_sel["acc"] - p_rand["acc"]
        per_seed.append({
            "seed": seed,
            "exclude_flagged": exclude_flagged,
            "overlap_with_flagged": overlap,
            "artifact": art_path,
            "manifest": man_path,
            "p_random_acc": p_rand["acc"],
            "p_random_minus_base": p_rand["acc"] - base_acc,
            "fixes": len(p_rand["fix_indices"]),
            "hurts": len(p_rand["hurt_indices"]),
            "changed": len(p_rand["changed_indices"]),
            "subset_base_acc": p_rand["subset_local"]["base_acc"],
            "subset_retry_acc": p_rand["subset_local"]["retry_acc"],
            "subset_delta": p_rand["subset_local"]["delta"],
            "targeting_marginal": targeting_marginal,
        })
        print(f"[seed={seed} excl_flagged={exclude_flagged} overlap={overlap}] "
              f"acc={p_rand['acc']*100:.2f}% (Δbase {p_rand['acc']*100 - base_acc*100:+.2f}pt) "
              f"subset {p_rand['subset_local']['base_acc']*100:.2f}%→{p_rand['subset_local']['retry_acc']*100:.2f}% "
              f"(Δ {p_rand['subset_local']['delta']*100:+.2f}pt) "
              f"targeting marginal {targeting_marginal*100:+.2f}pt")

    targeting_marginals = [p["targeting_marginal"] for p in per_seed]
    subset_deltas = [p["subset_delta"] for p in per_seed]
    p_random_accs = [p["p_random_acc"] for p in per_seed]
    m_tm, sd_tm = _mean_std(targeting_marginals)
    m_sd, sd_sd = _mean_std(subset_deltas)
    m_pr, sd_pr = _mean_std(p_random_accs)

    print(f"\n[aggregate across {len(per_seed)} random-control seeds]")
    print(f"  targeting marginal (P_selective − P_random): mean {m_tm*100:+.2f}pt ± {sd_tm*100:.2f}pt")
    print(f"  random subset-local retry delta:             mean {m_sd*100:+.2f}pt ± {sd_sd*100:.2f}pt")
    print(f"  P_random full-test:                          mean {m_pr*100:.2f}% ± {sd_pr*100:.2f}pt")
    n_pos_marginal = sum(1 for x in targeting_marginals if x > 0)
    n_neg_subset = sum(1 for x in subset_deltas if x < 0)
    print(f"  fraction of seeds with positive targeting marginal: {n_pos_marginal}/{len(per_seed)}")
    print(f"  fraction of seeds with NEGATIVE subset retry delta: {n_neg_subset}/{len(per_seed)}")

    os.makedirs(args.out_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    stem = args.output_stem or f"retry_control_multiseed_eval_{date_str}"

    out = {
        "config": {"answer_kind": args.answer_kind},
        "selective": {
            "artifact": args.selective_retry_artifact,
            "manifest": args.selective_manifest,
            "acc": p_sel["acc"],
            "fixes": len(p_sel["fix_indices"]),
            "hurts": len(p_sel["hurt_indices"]),
            "subset_local": p_sel["subset_local"],
        },
        "baseline_acc": base_acc,
        "per_seed": per_seed,
        "aggregate": {
            "targeting_marginal_mean": m_tm,
            "targeting_marginal_std": sd_tm,
            "subset_delta_mean": m_sd,
            "subset_delta_std": sd_sd,
            "p_random_mean": m_pr,
            "p_random_std": sd_pr,
            "n_pos_marginal": n_pos_marginal,
            "n_neg_subset_delta": n_neg_subset,
        },
    }
    json_path = os.path.join(args.out_dir, f"{stem}.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote: {json_path}")

    lines = []
    lines.append(f"# Phase E-Retry-Control-MultiSeed — {date_str}")
    lines.append("")
    lines.append(f"- selective retry artifact: `{args.selective_retry_artifact}`")
    lines.append(f"- selective manifest: `{args.selective_manifest}`")
    lines.append(f"- answer kind: `{args.answer_kind}`")
    lines.append(f"- random-control seeds: `{len(per_seed)}`")
    lines.append("")
    lines.append(f"baseline `exp_only` full-test: `{base_acc*100:.2f}%`; P_selective: `{p_sel['acc']*100:.2f}%` (`{(p_sel['acc']-base_acc)*100:+.2f}pt`)")
    lines.append("")
    lines.append("## Per-seed table")
    lines.append("")
    lines.append("| seed | exclude_flagged | overlap | P_random | Δbase | subset base | subset retry | subset Δ | targeting marginal |")
    lines.append("|---:|:---|---:|---:|---:|---:|---:|---:|---:|")
    for p in per_seed:
        lines.append(
            f"| {p['seed']} | {p['exclude_flagged']} | {p['overlap_with_flagged']} | "
            f"{p['p_random_acc']*100:.2f}% | `{p['p_random_minus_base']*100:+.2f}pt` | "
            f"{p['subset_base_acc']*100:.2f}% | {p['subset_retry_acc']*100:.2f}% | "
            f"`{p['subset_delta']*100:+.2f}pt` | **`{p['targeting_marginal']*100:+.2f}pt`** |"
        )
    lines.append("")
    lines.append("## Aggregate across random seeds")
    lines.append("")
    lines.append(f"- targeting marginal `(P_selective − P_random)`: mean `{m_tm*100:+.2f}pt` ± `{sd_tm*100:.2f}pt`")
    lines.append(f"- random subset-local retry delta: mean `{m_sd*100:+.2f}pt` ± `{sd_sd*100:.2f}pt`")
    lines.append(f"- fraction of seeds with positive targeting marginal: `{n_pos_marginal}/{len(per_seed)}`")
    lines.append(f"- fraction of seeds with negative random-subset retry delta: `{n_neg_subset}/{len(per_seed)}`")
    lines.append("")
    lines.append("## Reading guide")
    lines.append("")
    lines.append("- if targeting marginal mean is comfortably positive with a CI that excludes 0, the score-targeting claim survives single-seed noise on this artifact.")
    lines.append("- if all seeds have negative random subset-local retry delta, that pattern (\"retry on unflagged samples is net-negative\") is robust on this artifact, not a single-seed quirk.")
    lines.append("- a 95% CI on the mean is approximately `mean ± 1.96 · std / sqrt(K)`; with K=3 random-control seeds the CI is wide, so this is a robustness check more than a precise estimate.")
    md_path = os.path.join(args.out_dir, f"{stem}.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
