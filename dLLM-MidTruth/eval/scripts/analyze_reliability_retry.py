"""Phase E4: abstention-as-retry using existing cross-run answer pools.

This is an offline simulation, not a fresh generation run.

Idea
- Start from the same `exp_only` baseline used by Phase E/E2.
- Fit a per-sample reliability score on GSM8K validation data.
- Retry only the *least reliable* fraction of samples.
- A "retry" means swapping the baseline answer with an answer drawn from an
  already existing artifact (same-run cgap vote, prob vote, blockactive vote,
  final answer, or a small cross-run majority pool).

This tests whether the reliability score is useful not only for abstention, but
also for *targeted fallback* to an alternative answer source.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import analyze_reliability as e1  # noqa: E402
import analyze_reliability_v2 as e2  # noqa: E402


DEFAULT_GSM8K_BASE = e1.DEFAULT_GSM8K
DEFAULT_SVAMP_BASE = e1.DEFAULT_SVAMP
DEFAULT_GSM8K_PROB = os.path.join(
    REPO_ROOT,
    "outputs/LLaDA-8B-Instruct/20260430_cgap_answer_window5_prob_mean_rawsum_bs4_all_debug/"
    "gsm8k_gen128_steps64_vote_confidence_gap_answer_window5_prob_mean_rawsum_bs4/"
    "rank_0_generations.json",
)
DEFAULT_SVAMP_PROB = os.path.join(
    REPO_ROOT,
    "outputs/LLaDA-8B-Instruct/20260430_cgap_answer_window5_prob_mean_rawsum_bs4_all_debug/"
    "svamp_gen128_steps64_vote_confidence_gap_answer_window5_prob_mean_rawsum_bs4/"
    "rank_0_generations.json",
)
DEFAULT_GSM8K_BLOCKACTIVE = os.path.join(
    REPO_ROOT,
    "outputs/LLaDA-8B-Instruct/20260429_cgap_window5_blockactive_prob_mean_rawsum_bs4_all_debug/"
    "gsm8k_gen128_steps64_vote_confidence_gap_answer_window5_blockactive_prob_mean_rawsum_bs4/"
    "rank_0_generations.json",
)
DEFAULT_SVAMP_BLOCKACTIVE = os.path.join(
    REPO_ROOT,
    "outputs/LLaDA-8B-Instruct/20260429_cgap_window5_blockactive_prob_mean_rawsum_bs4_all_debug/"
    "svamp_gen128_steps64_vote_confidence_gap_answer_window5_blockactive_prob_mean_rawsum_bs4/"
    "rank_0_generations.json",
)
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "analysis")

RETRY_FRACS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def _load_answer_artifact(path):
    with open(path) as f:
        data = json.load(f)
    out = []
    for i, gen in enumerate(data["generations"]):
        valid = []
        for step in data["vote_debug"][i]["steps"]:
            if step.get("skip_reason") is not None:
                continue
            rw = step.get("raw_weight")
            pa = step.get("parsed_answer")
            if rw is None or pa is None:
                continue
            valid.append(
                {
                    "step": step["step"],
                    "total_steps": step["total_steps"],
                    "gap": float(rw),
                    "parsed_answer": pa,
                    "exp_weight": e1.math.exp(step["step"] / step["total_steps"] * e1.ALPHA),
                }
            )
        out.append(
            {
                "sample_index": i,
                "question": gen["question"],
                "ground_truth": gen["ground_truth"],
                "exp_only_answer": e1._exp_only_vote(valid),
                "stored_vote_answer": gen.get("vote_answer"),
                "stored_final_answer": gen.get("final_answer"),
            }
        )
    return out


def _key(row):
    return row["question"]


def _majority_vote(cands):
    cleaned = [c for c in cands if c is not None]
    if not cleaned:
        return None
    counts = Counter(cleaned)
    best_freq = max(counts.values())
    tied = [ans for ans, freq in counts.items() if freq == best_freq]
    if len(tied) == 1:
        return tied[0]
    # deterministic tie-break: keep first candidate order
    for c in cleaned:
        if c in tied:
            return c
    return cleaned[0]


def _merge_sources(base_feature_rows, base_answers, prob_answers, block_answers):
    by_q_prob = {_key(r): r for r in prob_answers}
    by_q_block = {_key(r): r for r in block_answers}
    by_q_base_ans = {_key(r): r for r in base_answers}
    merged = []
    for row in base_feature_rows:
        q = row["question"]
        b = by_q_base_ans[q]
        p = by_q_prob.get(q)
        blk = by_q_block.get(q)
        merged.append(
            {
                **row,
                "exp_only_answer": b["exp_only_answer"],
                "cgap_vote": b["stored_vote_answer"],
                "final_answer": b["stored_final_answer"],
                "prob_vote": None if p is None else p["stored_vote_answer"],
                "prob_final": None if p is None else p["stored_final_answer"],
                "blockactive_vote": None if blk is None else blk["stored_vote_answer"],
                "retry_majority": _majority_vote(
                    [
                        b["stored_vote_answer"],
                        None if p is None else p["stored_vote_answer"],
                        None if blk is None else blk["stored_vote_answer"],
                    ]
                ),
            }
        )
    return merged


def _load_feature_rows(path):
    feature_rows = e1._load(path)
    with open(path) as f:
        data = json.load(f)
    gens = data["generations"]
    assert len(feature_rows) == len(gens)
    out = []
    for row, gen in zip(feature_rows, gens):
        out.append({**row, "question": gen["question"], "ground_truth": gen["ground_truth"]})
    return out


def _fit_scores(gsm_rows):
    val, test = e1._split(gsm_rows)
    ys_val = [r["is_correct"] for r in val]
    val_corrs = {f: e1._pearson([r[f] for r in val], ys_val) for f in e1.FEATURES}

    zsum_spec = e1._fit_combined_score(val, e1.FEATURES)
    broad_features = list(zsum_spec.keys())
    inner_train, inner_dev = e2._inner_split(val)
    x_train, y_train, stats = e2._prepare_matrix(inner_train, broad_features)
    best = None
    for l2 in e2.L2_GRID:
        model = e2._fit_logreg_numpy(x_train, y_train, l2=l2)
        dev_scored = e2._score_rows(
            inner_dev,
            "model_score",
            e2._predict_scores(inner_dev, broad_features, stats, model),
        )
        aurc = e2._curve_pack(dev_scored, "model_score")["aurc"]
        if best is None or aurc < best["aurc"]:
            best = {"l2": l2, "aurc": aurc}

    x_full, y_full, full_stats = e2._prepare_matrix(val, broad_features)
    final_model = e2._fit_logreg_numpy(x_full, y_full, l2=best["l2"])

    def apply(rows):
        rows = [dict(r) for r in rows]
        e1._apply_combined_score(rows, zsum_spec)
        model_scores = e2._predict_scores(rows, broad_features, full_stats, final_model)
        for r, s in zip(rows, model_scores):
            r["logistic_broad_score"] = float(s)
        return rows

    return {
        "val": apply(val),
        "test": apply(test),
        "zsum_features": broad_features,
        "best_l2": best["l2"],
        "val_corrs": val_corrs,
    }


def _accuracy(rows, answer_key):
    n = len(rows)
    if n == 0:
        return 0.0
    hits = sum(1 for r in rows if e1._is_correct(r.get(answer_key), r["ground_truth"]))
    return hits / n


def _simulate_retry(rows, score_key, retry_answer_key, retry_frac):
    ordered = sorted(rows, key=lambda r: r[score_key])  # low score = retry first
    k = max(1, int(round(len(rows) * retry_frac)))
    retry_ids = set(r["sample_index"] for r in ordered[:k])
    out = []
    n_changed = 0
    n_fix = 0
    n_hurt = 0
    for r in rows:
        base = r["exp_only_answer"]
        cand = r.get(retry_answer_key)
        use = cand if r["sample_index"] in retry_ids and cand is not None else base
        if use != base:
            n_changed += 1
            if e1._is_correct(use, r["ground_truth"]) and not e1._is_correct(base, r["ground_truth"]):
                n_fix += 1
            if not e1._is_correct(use, r["ground_truth"]) and e1._is_correct(base, r["ground_truth"]):
                n_hurt += 1
        out.append(use)
    acc = sum(e1._is_correct(a, r["ground_truth"]) for a, r in zip(out, rows)) / len(rows)
    return {
        "retry_frac": retry_frac,
        "n_retry_target": k,
        "n_changed": n_changed,
        "acc": acc,
        "delta_vs_base": acc - _accuracy(rows, "exp_only_answer"),
        "fixes": n_fix,
        "hurts": n_hurt,
    }


def _sweep(rows, score_keys, retry_sources):
    out = {}
    for score_key in score_keys:
        out[score_key] = {}
        for src in retry_sources:
            trials = [_simulate_retry(rows, score_key, src, frac) for frac in RETRY_FRACS]
            best = max(trials, key=lambda x: x["acc"])
            out[score_key][src] = {"trials": trials, "best": best}
    return out


def _apply_best(rows, score_key, retry_answer_key, best_frac):
    return _simulate_retry(rows, score_key, retry_answer_key, best_frac)


def _oracle_pool(rows, answer_keys):
    hits = 0
    for r in rows:
        cands = [r["exp_only_answer"]] + [r.get(k) for k in answer_keys]
        ok = any(e1._is_correct(c, r["ground_truth"]) for c in cands if c is not None)
        hits += 1 if ok else 0
    return hits / len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gsm8k-base", default=DEFAULT_GSM8K_BASE)
    parser.add_argument("--svamp-base", default=DEFAULT_SVAMP_BASE)
    parser.add_argument("--gsm8k-prob", default=DEFAULT_GSM8K_PROB)
    parser.add_argument("--svamp-prob", default=DEFAULT_SVAMP_PROB)
    parser.add_argument("--gsm8k-block", default=DEFAULT_GSM8K_BLOCKACTIVE)
    parser.add_argument("--svamp-block", default=DEFAULT_SVAMP_BLOCKACTIVE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    gsm_feat = _load_feature_rows(args.gsm8k_base)
    gsm_base_ans = _load_answer_artifact(args.gsm8k_base)
    gsm_prob_ans = _load_answer_artifact(args.gsm8k_prob)
    gsm_block_ans = _load_answer_artifact(args.gsm8k_block)
    gsm_rows = _merge_sources(gsm_feat, gsm_base_ans, gsm_prob_ans, gsm_block_ans)

    svamp_feat = _load_feature_rows(args.svamp_base)
    svamp_base_ans = _load_answer_artifact(args.svamp_base)
    svamp_prob_ans = _load_answer_artifact(args.svamp_prob)
    svamp_block_ans = _load_answer_artifact(args.svamp_block)
    svamp_rows = _merge_sources(svamp_feat, svamp_base_ans, svamp_prob_ans, svamp_block_ans)

    split = _fit_scores(gsm_rows)
    gsm_val = split["val"]
    gsm_test = split["test"]

    score_keys = ["combined_score", "logistic_broad_score"]
    retry_sources = ["cgap_vote", "final_answer", "prob_vote", "blockactive_vote", "retry_majority"]

    val_sweep = _sweep(gsm_val, score_keys, retry_sources)

    chosen = {}
    for score_key in score_keys:
        chosen[score_key] = {}
        for src in retry_sources:
            chosen[score_key][src] = val_sweep[score_key][src]["best"]

    gsm_test_eval = {}
    svamp_eval = {}
    for score_key in score_keys:
        gsm_test_eval[score_key] = {}
        svamp_eval[score_key] = {}
        for src in retry_sources:
            frac = chosen[score_key][src]["retry_frac"]
            gsm_test_eval[score_key][src] = _apply_best(gsm_test, score_key, src, frac)
            # apply same val-fitted scores to svamp via refit logic on full GSM val
            svamp_scored = [dict(r) for r in svamp_rows]
            # reuse Phase E zsum
            if score_key == "combined_score":
                # rebuild from val spec by using fit_scores' side effect-free score rows is simpler:
                # use the same helper on the full svamp rows by borrowing from the val-fitted model
                # easiest path: derive from zsum features on each row already unavailable here, so
                # reuse the score pipeline via temporary fit on full merged rows done below.
                pass

    # Re-create SVAMP scores using the same GSM val-fitted specs.
    # combined_score
    zsum_spec = e1._fit_combined_score(gsm_val, e1.FEATURES)
    e1._apply_combined_score(svamp_rows, zsum_spec)
    # logistic_broad
    broad_features = split["zsum_features"]
    inner_train, inner_dev = e2._inner_split(gsm_val)
    x_train, y_train, stats = e2._prepare_matrix(inner_train, broad_features)
    best = None
    for l2 in e2.L2_GRID:
        model = e2._fit_logreg_numpy(x_train, y_train, l2=l2)
        dev_scored = e2._score_rows(
            inner_dev,
            "model_score",
            e2._predict_scores(inner_dev, broad_features, stats, model),
        )
        aurc = e2._curve_pack(dev_scored, "model_score")["aurc"]
        if best is None or aurc < best["aurc"]:
            best = {"l2": l2, "aurc": aurc}
    x_full, y_full, full_stats = e2._prepare_matrix(gsm_val, broad_features)
    final_model = e2._fit_logreg_numpy(x_full, y_full, l2=best["l2"])
    model_scores = e2._predict_scores(svamp_rows, broad_features, full_stats, final_model)
    for r, s in zip(svamp_rows, model_scores):
        r["logistic_broad_score"] = float(s)

    for score_key in score_keys:
        for src in retry_sources:
            frac = chosen[score_key][src]["retry_frac"]
            svamp_eval.setdefault(score_key, {})[src] = _apply_best(svamp_rows, score_key, src, frac)

    base_acc = {
        "gsm8k_val": _accuracy(gsm_val, "exp_only_answer"),
        "gsm8k_test": _accuracy(gsm_test, "exp_only_answer"),
        "svamp_full": _accuracy(svamp_rows, "exp_only_answer"),
    }
    standalone = {
        "gsm8k_test": {k: _accuracy(gsm_test, k) for k in retry_sources},
        "svamp_full": {k: _accuracy(svamp_rows, k) for k in retry_sources},
    }

    oracle_pool = {
        "gsm8k_test": _oracle_pool(gsm_test, retry_sources),
        "svamp_full": _oracle_pool(svamp_rows, retry_sources),
    }

    date_str = datetime.now().strftime("%Y%m%d")
    stem = f"reliability_retry_{date_str}"
    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, stem + ".json")
    md_path = os.path.join(args.out_dir, stem + ".md")

    out = {
        "config": {
            "seed": e1.SEED,
            "validation_frac": e1.VALIDATION_FRAC,
            "retry_fracs": RETRY_FRACS,
            "score_keys": score_keys,
            "retry_sources": retry_sources,
        },
        "runs": {
            "gsm8k_base": args.gsm8k_base,
            "svamp_base": args.svamp_base,
            "gsm8k_prob": args.gsm8k_prob,
            "svamp_prob": args.svamp_prob,
            "gsm8k_block": args.gsm8k_block,
            "svamp_block": args.svamp_block,
        },
        "base_acc": base_acc,
        "standalone_source_acc": standalone,
        "oracle_pool_acc": oracle_pool,
        "val_sweep": val_sweep,
        "chosen": chosen,
        "gsm8k_test": gsm_test_eval,
        "svamp": svamp_eval,
    }
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    lines = []
    lines.append(f"# Reliability Retry Sweep — {date_str}")
    lines.append("")
    lines.append("Offline E4 simulation: retry only the least reliable samples and swap their baseline `exp_only` answer with an answer from an already existing artifact/source.")
    lines.append("")
    lines.append("## Base accuracy")
    lines.append("")
    lines.append(f"- GSM8K val:  {base_acc['gsm8k_val']:.2%}")
    lines.append(f"- GSM8K test: {base_acc['gsm8k_test']:.2%}")
    lines.append(f"- SVAMP full: {base_acc['svamp_full']:.2%}")
    lines.append("")
    lines.append("## Standalone retry-source accuracy")
    lines.append("")
    lines.append("| Source | GSM8K test | SVAMP |")
    lines.append("|---|---:|---:|")
    for src in retry_sources:
        lines.append(f"| {src} | {standalone['gsm8k_test'][src]:.2%} | {standalone['svamp_full'][src]:.2%} |")
    lines.append(f"| oracle_pool | {oracle_pool['gsm8k_test']:.2%} | {oracle_pool['svamp_full']:.2%} |")
    lines.append("")
    for score_key in score_keys:
        lines.append(f"## {score_key} — val-selected retry budget")
        lines.append("")
        lines.append("| Retry source | Best retry frac (val) | Val acc | Test acc | Test delta | SVAMP acc | SVAMP delta |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for src in retry_sources:
            best = chosen[score_key][src]
            test = gsm_test_eval[score_key][src]
            sv = svamp_eval[score_key][src]
            lines.append(
                f"| {src} | {best['retry_frac']:.0%} | {best['acc']:.2%} | {test['acc']:.2%} | {test['delta_vs_base']:+.2%} | {sv['acc']:.2%} | {sv['delta_vs_base']:+.2%} |"
            )
        lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- This is not a fresh retry generation; it is an offline substitution test using existing answer sources.")
    lines.append("- The baseline answer is reconstructed `exp_only` from the current debug artifact, matching Phase E/E2 labels.")
    lines.append("- A positive result would mean the reliability score is not only abstention-useful, but also can target samples that benefit from switching answer sources.")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
