"""Phase 1 — Within-artifact candidate-set oracle ceiling diagnostic.

Methodology-clean candidate-answer reranking probe. From a SINGLE T=0 base
artifact, derive multiple candidate answers per sample, collapse to unique
answers, and measure oracle ceilings under three lenses.

Setup constraints (T=0, parser fixed, baseline-comparable):
- single base artifact: GSM8K everpass cgap_logit debug
- no rerun, no cross-artifact mixing in Phase 1 (cross-artifact is Phase 1b)
- candidates derived entirely from `vote_debug[i]["steps"]` and the stored
  generation/vote fields of that one artifact

Candidates (8):
  1. exp_only                       — offline temporal exp vote
  2. native_vote_answer             — stored vote_answer (artifact's native vote)
  3. final_answer                   — stored final-state parse
  4. last_valid_answer              — last non-skipped step's parsed_answer
  5. late_window_majority_q25       — majority over steps with step_idx >= 0.75 * total
  6. late_window_majority_q50       — majority over steps with step_idx >= 0.50 * total
  7. longest_run_answer             — answer with longest contiguous run in valid steps
  8. most_persistent_answer         — answer with the most total valid steps

Oracles:
  A. full unique-answer oracle      — any candidate equals gt
  B. disagreement-only oracle       — restrict to samples where ≥2 unique candidates
                                       exist; oracle on this subset
  C. provenance-restricted:
      compact = {exp_only, native_vote, final_answer}
      full    = compact + temporal (4..8)
      diff = full_oracle − compact_oracle  (the temporal-candidates marginal headroom)

Termination criteria:
- Option A (terminate raw-acc line): full oracle ≤ 70% OR temporal marginal ≤ 1pt
- Option B (proceed to Phase 2): full vs compact gain ≥ 2pt OR overall full ≥ 72%
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import analyze_reliability as e1  # noqa: E402

DEFAULT_BASE_ARTIFACT = os.path.join(
    REPO_ROOT,
    "outputs/LLaDA-8B-Instruct/20260429_everpass_debug_bs4/"
    "gsm8k_gen128_steps64_vote_confidence_gap_answer_window5_mean_rawsum_bs4_debug/"
    "rank_0_generations.json",
)
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "analysis")
ALPHA = 5.0


def _is_correct(pred, gt, atol=1e-6):
    if pred is None or gt is None:
        return False
    try:
        return abs(float(pred) - float(gt)) < atol
    except (TypeError, ValueError):
        return str(pred) == str(gt)


def _exp_only_vote(valid_events):
    if not valid_events:
        return None
    scores = {}
    for ev in valid_events:
        scores[ev["parsed_answer"]] = scores.get(ev["parsed_answer"], 0.0) + ev["exp_weight"]
    if not scores:
        return None
    return max(scores.items(), key=lambda kv: kv[1])[0]


def _normalize_answer_key(ans):
    """Use a canonical key so 18.0 and 18 collapse to the same candidate."""
    if ans is None:
        return None
    try:
        f = float(ans)
        if f.is_integer():
            return int(f)
        return f
    except (TypeError, ValueError):
        return str(ans)


def _derive_candidates(artifact_generation_i, vote_debug_i):
    """Compute 8 candidate answers for a single sample."""
    steps = vote_debug_i["steps"]
    total_steps = vote_debug_i["steps"][0]["total_steps"] if steps else 64

    # Build valid event list (skipped steps excluded)
    valid = []
    for step in steps:
        sr = step.get("skip_reason")
        if sr is not None:
            continue
        rw = step.get("raw_weight")
        pa = step.get("parsed_answer")
        if rw is None or pa is None:
            continue
        s = step["step"]
        ts = step["total_steps"]
        valid.append({
            "step": s,
            "total_steps": ts,
            "parsed_answer": pa,
            "gap": float(rw),
            "exp_weight": math.exp(s / ts * ALPHA),
        })

    # 1. exp_only
    exp_only = _exp_only_vote(valid)

    # 2. native vote_answer (stored)
    native = artifact_generation_i.get("vote_answer")

    # 3. final_answer (stored)
    final_ans = artifact_generation_i.get("final_answer")

    # 4. last_valid_answer (last non-skipped step)
    last_valid = valid[-1]["parsed_answer"] if valid else None

    # 5/6. late_window_majority_q25 / q50
    def _late_majority(threshold_frac):
        late = [ev for ev in valid if ev["step"] >= threshold_frac * total_steps]
        if not late:
            return None
        cnts = Counter([ev["parsed_answer"] for ev in late])
        max_c = max(cnts.values())
        tied = [a for a, c in cnts.items() if c == max_c]
        # first-occurrence tie-break by latest step appearance? use last-occurrence as more natural for "late window"
        for ev in reversed(late):
            if ev["parsed_answer"] in tied:
                return ev["parsed_answer"]
        return tied[0]

    late_q25 = _late_majority(0.75)
    late_q50 = _late_majority(0.50)

    # 7. longest_run_answer (contiguous runs in valid sequence)
    longest_run = None
    longest_len = 0
    if valid:
        cur_ans = valid[0]["parsed_answer"]
        cur_len = 1
        longest_run = cur_ans
        longest_len = 1
        for ev in valid[1:]:
            if ev["parsed_answer"] == cur_ans:
                cur_len += 1
            else:
                cur_ans = ev["parsed_answer"]
                cur_len = 1
            if cur_len > longest_len:
                longest_len = cur_len
                longest_run = cur_ans

    # 8. most_persistent_answer
    if valid:
        cnts = Counter([ev["parsed_answer"] for ev in valid])
        most_persistent = max(cnts.items(), key=lambda kv: kv[1])[0]
    else:
        most_persistent = None

    # Extra trajectory-state features (for Phase 2 use; not candidates)
    final_run_length = 0
    final_run_ans = None
    if valid:
        final_run_ans = valid[-1]["parsed_answer"]
        for ev in reversed(valid):
            if ev["parsed_answer"] == final_run_ans:
                final_run_length += 1
            else:
                break
    final_run_step_frac = (valid[-1]["step"] / total_steps) if valid else 0.0
    last_change_step_frac = None
    if valid:
        prev = object()
        last_change = valid[0]
        for ev in valid:
            if ev["parsed_answer"] != prev:
                last_change = ev
                prev = ev["parsed_answer"]
        last_change_step_frac = last_change["step"] / total_steps

    return {
        "candidates_named": {
            "exp_only": exp_only,
            "native_vote_answer": native,
            "final_answer": final_ans,
            "last_valid_answer": last_valid,
            "late_window_majority_q25": late_q25,
            "late_window_majority_q50": late_q50,
            "longest_run_answer": longest_run,
            "most_persistent_answer": most_persistent,
        },
        "features": {
            "n_valid": len(valid),
            "final_run_length": final_run_length,
            "final_run_step_frac": final_run_step_frac,
            "last_change_step_frac": last_change_step_frac,
            "n_unique_answers": len(set(ev["parsed_answer"] for ev in valid)) if valid else 0,
        },
    }


def _unique_answer_set(candidates_named):
    """Collapse readouts → unique answer candidates with their supporting readouts."""
    out = {}
    for readout, ans in candidates_named.items():
        if ans is None:
            continue
        key = _normalize_answer_key(ans)
        if key not in out:
            out[key] = {"answer": ans, "key": key, "readouts": []}
        out[key]["readouts"].append(readout)
    return list(out.values())


def _split(rows, frac=0.8, seed=42):
    import random
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    n_val = int(len(rows) * frac)
    val_ids = set(idx[:n_val])
    val = [rows[i] for i in range(len(rows)) if i in val_ids]
    test = [rows[i] for i in range(len(rows)) if i not in val_ids]
    return val, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-artifact", default=DEFAULT_BASE_ARTIFACT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--output-stem", default="")
    args = parser.parse_args()

    print(f"[load] {args.base_artifact}")
    with open(args.base_artifact) as f:
        data = json.load(f)
    n = len(data["generations"])
    print(f"  n_samples = {n}")
    print(f"  stored vote_method: {data.get('vote_method')}")
    print(f"  batch_size: {data.get('batch_size')}  temperature: {data.get('temperature')}")

    # Derive candidates per sample
    per_sample = []
    for i, gen in enumerate(data["generations"]):
        gt = gen.get("ground_truth")
        candidates = _derive_candidates(gen, data["vote_debug"][i])
        unique_answers = _unique_answer_set(candidates["candidates_named"])
        # Mark which unique answer is correct
        for ua in unique_answers:
            ua["correct"] = int(_is_correct(ua["answer"], gt))
        per_sample.append({
            "sample_index": i,
            "ground_truth": gt,
            "candidates_named": candidates["candidates_named"],
            "candidate_correct": {
                name: int(_is_correct(ans, gt)) for name, ans in candidates["candidates_named"].items()
            },
            "unique_answers": unique_answers,
            "features": candidates["features"],
        })

    val, test = _split(per_sample, frac=0.8, seed=42)
    print(f"[split] val n={len(val)}  test n={len(test)} (seed=42, frac=0.8)")

    # === Diagnostics: per-readout accuracy on val and test ===
    readout_names = list(per_sample[0]["candidates_named"].keys())
    print(f"\n[per-readout accuracy: val (n={len(val)}) and test (n={len(test)})]")
    print(f"{'readout':<30}  {'val':>10}  {'test':>10}")
    readout_val_acc = {}
    readout_test_acc = {}
    for r in readout_names:
        val_c = sum(s["candidate_correct"][r] for s in val)
        test_c = sum(s["candidate_correct"][r] for s in test)
        readout_val_acc[r] = val_c / len(val)
        readout_test_acc[r] = test_c / len(test)
        print(f"  {r:<30}  {readout_val_acc[r]*100:>8.2f}%  {readout_test_acc[r]*100:>8.2f}%")

    # === Val-selected best readout, one-shot test eval ===
    val_best = max(readout_val_acc.items(), key=lambda kv: kv[1])
    val_best_name = val_best[0]
    val_best_test_acc = readout_test_acc[val_best_name]
    val_exp_only = readout_val_acc["exp_only"]
    test_exp_only = readout_test_acc["exp_only"]
    print(f"\n[val-selected single-readout deployment]")
    print(f"  val best readout: {val_best_name}  (val_acc={val_best[1]*100:.2f}%)")
    print(f"  → test acc (one-shot): {val_best_test_acc*100:.2f}%  (vs exp_only test {test_exp_only*100:.2f}%)")
    print(f"  delta over exp_only on test: {(val_best_test_acc - test_exp_only)*100:+.2f}pt")

    # === Oracle A: full unique-answer oracle on test ===
    n_test = len(test)
    full_oracle_correct = sum(
        1 for s in test if any(ua["correct"] for ua in s["unique_answers"])
    )
    full_oracle_acc = full_oracle_correct / n_test
    print(f"\n[Oracle A — full unique-answer oracle, test]")
    print(f"  acc: {full_oracle_acc*100:.2f}%  ({full_oracle_correct}/{n_test})")

    # === Oracle C: provenance-restricted ===
    compact_readouts = ["exp_only", "native_vote_answer", "final_answer"]
    full_readouts = readout_names  # all 8

    def _oracle_under(test_samples, allowed_readouts):
        correct = 0
        for s in test_samples:
            allowed_answers = {_normalize_answer_key(s["candidates_named"][r]) for r in allowed_readouts}
            allowed_answers.discard(None)
            for ua in s["unique_answers"]:
                if ua["key"] in allowed_answers and ua["correct"]:
                    correct += 1
                    break
        return correct, correct / len(test_samples)

    compact_correct, compact_acc = _oracle_under(test, compact_readouts)
    full_correct, full_acc = _oracle_under(test, full_readouts)
    temporal_marginal = full_acc - compact_acc
    print(f"\n[Oracle C — provenance-restricted, test]")
    print(f"  compact (exp/native/final): {compact_acc*100:.2f}% ({compact_correct}/{n_test})")
    print(f"  full (+ temporal 4..8):     {full_acc*100:.2f}% ({full_correct}/{n_test})")
    print(f"  temporal marginal:          {temporal_marginal*100:+.2f}pt")

    # === Oracle B: disagreement-only oracle (subset where ≥2 unique candidates) ===
    disagreement_subset = [s for s in test if len(s["unique_answers"]) >= 2]
    n_dis = len(disagreement_subset)
    if n_dis:
        dis_oracle_correct = sum(
            1 for s in disagreement_subset if any(ua["correct"] for ua in s["unique_answers"])
        )
        dis_oracle_acc = dis_oracle_correct / n_dis
        # On disagreement subset, what's exp_only acc?
        dis_exp_correct = sum(s["candidate_correct"]["exp_only"] for s in disagreement_subset)
        dis_exp_acc = dis_exp_correct / n_dis
        dis_marginal = dis_oracle_acc - dis_exp_acc
    else:
        dis_oracle_correct = dis_oracle_acc = 0
        dis_exp_correct = dis_exp_acc = 0
        dis_marginal = 0
    print(f"\n[Oracle B — disagreement-only oracle, test]")
    print(f"  disagreement subset (≥2 unique): {n_dis}/{n_test} samples ({n_dis/n_test*100:.2f}%)")
    print(f"  exp_only acc on subset:        {dis_exp_acc*100:.2f}%  ({dis_exp_correct}/{n_dis})")
    print(f"  oracle acc on subset:          {dis_oracle_acc*100:.2f}%  ({dis_oracle_correct}/{n_dis})")
    print(f"  subset oracle marginal:        {dis_marginal*100:+.2f}pt")
    # Translated to full-test gain (if we could perfectly route disagreement samples)
    full_test_gain = (dis_oracle_correct - dis_exp_correct) / n_test
    print(f"  full-test gain if perfect routing on disagreement subset: {full_test_gain*100:+.2f}pt")

    # === Distribution of unique-answer counts ===
    uniq_count_hist = Counter(len(s["unique_answers"]) for s in test)
    print(f"\n[unique-answer count distribution, test]")
    for k in sorted(uniq_count_hist.keys()):
        print(f"  {k} unique answers: {uniq_count_hist[k]} samples")

    # === Decision under termination criteria ===
    exp_baseline = readout_test_acc["exp_only"]
    print(f"\n[decision]")
    print(f"  baseline exp_only test:       {exp_baseline*100:.2f}%")
    print(f"  full oracle test:             {full_acc*100:.2f}% (potential ceiling)")
    print(f"  temporal marginal vs compact: {temporal_marginal*100:+.2f}pt")
    option_a_terminate = (full_acc <= 0.70) or (temporal_marginal <= 0.01)
    option_b_proceed = (temporal_marginal >= 0.02) or (full_acc >= 0.72)
    print(f"  Option A (terminate): full ≤ 70% or temporal marginal ≤ 1pt → {option_a_terminate}")
    print(f"  Option B (proceed):   full vs compact gain ≥ 2pt or full ≥ 72% → {option_b_proceed}")

    # === Write outputs ===
    os.makedirs(args.out_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    stem = args.output_stem or f"router_oracle_phase1_{date_str}"

    out = {
        "config": {
            "base_artifact": args.base_artifact,
            "n_total": n,
            "n_val": len(val),
            "n_test": len(test),
            "outer_split_seed": 42,
            "outer_split_frac": 0.8,
        },
        "readout_val_acc": readout_val_acc,
        "readout_test_acc": readout_test_acc,
        "val_selected_single_readout": {
            "name": val_best_name,
            "val_acc": readout_val_acc[val_best_name],
            "test_acc": val_best_test_acc,
            "delta_vs_exp_only_test": val_best_test_acc - test_exp_only,
        },
        "oracle_A_full": {"correct": full_oracle_correct, "n": n_test, "acc": full_oracle_acc},
        "oracle_B_disagreement_only": {
            "n_subset": n_dis, "exp_acc": dis_exp_acc, "oracle_acc": dis_oracle_acc,
            "subset_marginal": dis_marginal, "full_test_gain_if_perfect": full_test_gain,
        },
        "oracle_C_provenance": {
            "compact_readouts": compact_readouts,
            "full_readouts": full_readouts,
            "compact_acc": compact_acc, "full_acc": full_acc,
            "temporal_marginal": temporal_marginal,
        },
        "unique_count_distribution": dict(uniq_count_hist),
        "decision": {
            "option_a_terminate": option_a_terminate,
            "option_b_proceed": option_b_proceed,
            "full_oracle_acc": full_acc,
            "temporal_marginal_pt": temporal_marginal,
        },
        "per_sample_test": [
            {
                "sample_index": s["sample_index"],
                "ground_truth": s["ground_truth"],
                "candidates_named": s["candidates_named"],
                "candidate_correct": s["candidate_correct"],
                "unique_answers": [
                    {"answer": ua["answer"], "readouts": ua["readouts"], "correct": ua["correct"]}
                    for ua in s["unique_answers"]
                ],
                "features": s["features"],
            }
            for s in test
        ],
        "per_sample_val": [
            {
                "sample_index": s["sample_index"],
                "ground_truth": s["ground_truth"],
                "candidates_named": s["candidates_named"],
                "candidate_correct": s["candidate_correct"],
                "unique_answers": [
                    {"answer": ua["answer"], "readouts": ua["readouts"], "correct": ua["correct"]}
                    for ua in s["unique_answers"]
                ],
                "features": s["features"],
            }
            for s in val
        ],
    }
    json_path = os.path.join(args.out_dir, f"{stem}.json")
    md_path = os.path.join(args.out_dir, f"{stem}.md")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote: {json_path}")

    # === Markdown ===
    lines = []
    lines.append(f"# Phase 1 — Within-artifact Oracle Ceiling Diagnostic — {date_str}")
    lines.append("")
    lines.append("Methodology-clean probe: single T=0 base artifact, 8 candidate answers per sample, oracle ceilings under 3 lenses. Decision criteria predefined.")
    lines.append("")
    lines.append(f"- base artifact: `{args.base_artifact}`")
    lines.append(f"- stored vote_method: `{data.get('vote_method')}`")
    lines.append(f"- batch_size: `{data.get('batch_size')}`  temperature: `{data.get('temperature')}`")
    lines.append(f"- total samples: `{n}`  test split (seed=42, frac=0.8): `{n_test}`")
    lines.append("")
    lines.append("## Per-readout accuracy (val + test)")
    lines.append("")
    lines.append("| Readout | Val Acc | Test Acc |")
    lines.append("|---|---:|---:|")
    for r in readout_names:
        va = readout_val_acc[r]
        ta = readout_test_acc[r]
        marker = " ← baseline" if r == "exp_only" else ""
        lines.append(f"| {r}{marker} | {va*100:.2f}% | {ta*100:.2f}% |")
    lines.append("")
    lines.append("## Val-selected single-readout deployment (one-shot test)")
    lines.append("")
    lines.append(f"- val-best readout: **`{val_best_name}`** (val_acc `{readout_val_acc[val_best_name]*100:.2f}%`)")
    lines.append(f"- test acc under this readout: **`{val_best_test_acc*100:.2f}%`**")
    lines.append(f"- delta vs exp_only on test: `{(val_best_test_acc - test_exp_only)*100:+.2f}pt`")
    lines.append("")
    lines.append("This is the simplest possible improvement: pick a single deterministic aggregator on val, deploy on test. No routing, no learned scorer, no rerun.")
    lines.append("")
    lines.append("## Oracle A — full unique-answer ceiling")
    lines.append("")
    lines.append(f"- {full_oracle_correct}/{n_test} samples have at least one correct candidate")
    lines.append(f"- **Oracle A acc: `{full_oracle_acc*100:.2f}%`** (vs exp_only baseline `{exp_baseline*100:.2f}%`)")
    lines.append(f"- Ceiling headroom: `{(full_oracle_acc - exp_baseline)*100:+.2f}pt`")
    lines.append("")
    lines.append("## Oracle B — disagreement-only ceiling")
    lines.append("")
    lines.append(f"- Disagreement subset: `{n_dis}/{n_test}` samples with ≥2 unique candidates")
    lines.append(f"- exp_only on subset: `{dis_exp_acc*100:.2f}%` ({dis_exp_correct}/{n_dis})")
    lines.append(f"- Oracle on subset: `{dis_oracle_acc*100:.2f}%` ({dis_oracle_correct}/{n_dis})")
    lines.append(f"- Subset oracle marginal: `{dis_marginal*100:+.2f}pt`")
    lines.append(f"- Full-test gain if perfect routing on this subset: `{full_test_gain*100:+.2f}pt`")
    lines.append("")
    lines.append("## Oracle C — provenance-restricted")
    lines.append("")
    lines.append("| Set | Readouts | Oracle acc |")
    lines.append("|---|---|---:|")
    lines.append(f"| compact | exp_only + native_vote_answer + final_answer | `{compact_acc*100:.2f}%` |")
    lines.append(f"| full | compact + 5 temporal candidates | `{full_acc*100:.2f}%` |")
    lines.append("")
    lines.append(f"**Temporal-candidates marginal headroom: `{temporal_marginal*100:+.2f}pt`**")
    lines.append("")
    lines.append("This tells us whether the temporal persistence decoder adds genuine new headroom beyond compact 3-readout set.")
    lines.append("")
    lines.append("## Unique-answer count distribution (test)")
    lines.append("")
    lines.append("| # unique candidates | # samples |")
    lines.append("|---:|---:|")
    for k in sorted(uniq_count_hist.keys()):
        lines.append(f"| {k} | {uniq_count_hist[k]} |")
    lines.append("")
    lines.append("## Decision under predefined termination criteria")
    lines.append("")
    lines.append("- Option A (terminate raw-acc line): full oracle ≤ 70% **OR** temporal marginal ≤ 1pt")
    lines.append("- Option B (proceed to Phase 2): full vs compact gain ≥ 2pt **OR** full oracle ≥ 72%")
    lines.append("")
    lines.append(f"- full oracle: `{full_acc*100:.2f}%`")
    lines.append(f"- temporal marginal: `{temporal_marginal*100:+.2f}pt`")
    lines.append(f"- **Option A triggered: `{option_a_terminate}`**")
    lines.append(f"- **Option B triggered: `{option_b_proceed}`**")
    lines.append("")
    if option_b_proceed and not option_a_terminate:
        lines.append("**Recommendation**: proceed to Phase 2 (B1 conservative answer-coalition override first).")
    elif option_a_terminate:
        lines.append("**Recommendation**: terminate raw-accuracy line; cgap project's accuracy improvement budget is exhausted at this baseline.")
    else:
        lines.append("**Recommendation**: ambiguous — neither criterion cleanly triggered. Inspect manually.")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
