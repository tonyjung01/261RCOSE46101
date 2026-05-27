"""Parser-side diagnosis for Math500 Bucket A events.

Bucket A was defined in Phase 2 as AWNF events where the span-aware parser fails
to recover any usable `(answer, char_start, char_end)` span. This script breaks
Bucket A down into coarse text-pattern classes to help distinguish:

- missing answer formatting entirely
- malformed boxed-like output
- strongly corrupted "boxedboxed" style output

Outputs both markdown and JSON under `eval/analysis/`.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from utils.span_parsers import parse_math_answer_with_span  # noqa: E402


DEFAULT_RUN = os.path.join(
    REPO_ROOT,
    "outputs/LLaDA-8B-Instruct/20260430_cgap_answer_window5_prob_mean_rawsum_bs4_all_debug/"
    "math500_gen128_steps64_vote_confidence_gap_answer_window5_prob_mean_rawsum_bs4/"
    "rank_0_generations.json",
)
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "analysis")

CATEGORY_ORDER = [
    "boxedboxed_corruption",
    "boxed_like_but_unparseable",
    "no_boxed_but_answer_tag",
    "no_boxed_no_answer_tag",
]

STEP_BINS = [
    ("Q1 (0-15)", 0, 16),
    ("Q2 (16-31)", 16, 32),
    ("Q3 (32-47)", 32, 48),
    ("Q4 (48-63)", 48, 64),
]


def classify_bucket_a(text: str) -> str:
    lowered = text.lower()
    has_boxed = "boxed" in lowered
    has_boxedboxed = "boxedboxed" in lowered
    has_answer_tag = "<answer>" in lowered or "</answer>" in lowered

    if has_boxedboxed:
        return "boxedboxed_corruption"
    if has_boxed:
        return "boxed_like_but_unparseable"
    if has_answer_tag:
        return "no_boxed_but_answer_tag"
    return "no_boxed_no_answer_tag"


def bin_name(step: int) -> str:
    for name, lo, hi in STEP_BINS:
        if lo <= step < hi:
            return name
    return "other"


def short_snippet(text: str, limit: int = 220) -> str:
    text = text.replace("\n", "\\n")
    return text[:limit]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--task", default="math500")
    args = parser.parse_args()

    with open(args.run) as f:
        data = json.load(f)

    category_counts = Counter()
    feature_counts = Counter()
    category_step_counts = defaultdict(Counter)
    per_sample = defaultdict(Counter)
    examples = {}

    total_bucket_a = 0
    total_awnf = 0

    for sample_idx, vd in enumerate(data["vote_debug"]):
        for step in vd["steps"]:
            if step.get("skip_reason") != "answer_window_not_found":
                continue
            total_awnf += 1
            text = step.get("parsed_answer", "")
            _, char_start, _ = parse_math_answer_with_span(text)
            if char_start >= 0:
                continue

            total_bucket_a += 1
            category = classify_bucket_a(text)
            category_counts[category] += 1
            category_step_counts[category][bin_name(step["step"])] += 1
            per_sample[sample_idx][category] += 1

            lowered = text.lower()
            feature_counts["has_boxed"] += "boxed" in lowered
            feature_counts["has_boxed_open"] += "\\boxed{" in text or "boxed{" in text
            feature_counts["has_boxedboxed"] += "boxedboxed" in lowered
            feature_counts["has_answer_tag"] += "<answer>" in lowered or "</answer>" in lowered
            feature_counts["has_frac"] += "\\frac" in text
            feature_counts["has_digit"] += any(ch.isdigit() for ch in text)
            feature_counts["len_gt_80"] += len(text) > 80
            feature_counts["len_gt_200"] += len(text) > 200

            if category not in examples:
                examples[category] = {
                    "sample": sample_idx,
                    "step": step["step"],
                    "snippet": short_snippet(text),
                }

    events_per_sample = [sum(c.values()) for c in per_sample.values()]
    dominant_category_samples = Counter(c.most_common(1)[0][0] for c in per_sample.values()) if per_sample else Counter()

    report = {
        "run": args.run,
        "task": args.task,
        "total_awnf": total_awnf,
        "total_bucket_a": total_bucket_a,
        "bucket_a_share_of_awnf": (total_bucket_a / total_awnf) if total_awnf else 0.0,
        "category_counts": dict(category_counts),
        "category_step_counts": {k: dict(v) for k, v in category_step_counts.items()},
        "feature_counts": dict(feature_counts),
        "samples_with_bucket_a": len(per_sample),
        "events_per_sample_mean": statistics.mean(events_per_sample) if events_per_sample else 0.0,
        "events_per_sample_median": statistics.median(events_per_sample) if events_per_sample else 0.0,
        "samples_with_ge_32_events": sum(x >= 32 for x in events_per_sample),
        "samples_with_ge_48_events": sum(x >= 48 for x in events_per_sample),
        "dominant_category_samples": dict(dominant_category_samples),
        "examples": examples,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    json_path = os.path.join(args.out_dir, f"{args.task}_bucket_a_{date_str}.json")
    md_path = os.path.join(args.out_dir, f"{args.task}_bucket_a_{date_str}.md")

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append(f"# {args.task} Bucket A Parser-Side Diagnosis — {date_str}")
    lines.append("")
    lines.append(f"Run: `{args.run}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total AWNF events: `{total_awnf}`")
    lines.append(f"- Total Bucket A events: `{total_bucket_a}` ({(total_bucket_a / total_awnf * 100) if total_awnf else 0:.1f}% of AWNF)")
    lines.append(f"- Samples affected by Bucket A: `{len(per_sample)}` / `{len(data['vote_debug'])}`")
    lines.append(f"- Mean Bucket A events per affected sample: `{report['events_per_sample_mean']:.2f}`")
    lines.append(f"- Median Bucket A events per affected sample: `{report['events_per_sample_median']:.0f}`")
    lines.append(f"- Samples with >=32 Bucket A events: `{report['samples_with_ge_32_events']}`")
    lines.append(f"- Samples with >=48 Bucket A events: `{report['samples_with_ge_48_events']}`")
    lines.append("")
    lines.append("## Category breakdown")
    lines.append("")
    lines.append("| Category | Count | % of Bucket A | Example interpretation |")
    lines.append("|---|---:|---:|---|")
    interpretations = {
        "boxedboxed_corruption": "repeated/corrupted `boxedboxed...` style output",
        "boxed_like_but_unparseable": "contains boxed-like text but no recoverable valid span",
        "no_boxed_but_answer_tag": "answer-section style text with no boxed answer",
        "no_boxed_no_answer_tag": "no boxed marker and no answer tag; likely unfinished or off-format generation",
    }
    for cat in CATEGORY_ORDER:
        count = category_counts[cat]
        pct = count / total_bucket_a * 100 if total_bucket_a else 0.0
        lines.append(f"| {cat} | {count} | {pct:.1f}% | {interpretations[cat]} |")
    lines.append("")
    lines.append("## Feature counts")
    lines.append("")
    lines.append("| Feature | Count | % of Bucket A |")
    lines.append("|---|---:|---:|")
    for feat in [
        "has_boxed",
        "has_boxed_open",
        "has_boxedboxed",
        "has_answer_tag",
        "has_frac",
        "has_digit",
        "len_gt_80",
        "len_gt_200",
    ]:
        count = feature_counts[feat]
        pct = count / total_bucket_a * 100 if total_bucket_a else 0.0
        lines.append(f"| {feat} | {count} | {pct:.1f}% |")
    lines.append("")
    lines.append("## Step-bin profile")
    lines.append("")
    lines.append("| Category | Q1 (0-15) | Q2 (16-31) | Q3 (32-47) | Q4 (48-63) |")
    lines.append("|---|---:|---:|---:|---:|")
    for cat in CATEGORY_ORDER:
        counts = category_step_counts[cat]
        lines.append(
            f"| {cat} | {counts['Q1 (0-15)']} | {counts['Q2 (16-31)']} | {counts['Q3 (32-47)']} | {counts['Q4 (48-63)']} |"
        )
    lines.append("")
    lines.append("## Dominant category by affected sample")
    lines.append("")
    lines.append("| Dominant category | Sample count |")
    lines.append("|---|---:|")
    for cat, count in dominant_category_samples.most_common():
        lines.append(f"| {cat} | {count} |")
    lines.append("")
    lines.append("## Example snippets")
    lines.append("")
    for cat in CATEGORY_ORDER:
        ex = examples.get(cat)
        if not ex:
            continue
        lines.append(f"- `{cat}` — sample `{ex['sample']}`, step `{ex['step']}`")
        lines.append(f"  - `{ex['snippet']}`")
    lines.append("")
    lines.append("## Tentative read")
    lines.append("")
    lines.append("- If `no_boxed_*` categories dominate, the main issue is likely not token alignment but the model never producing a recoverable boxed answer span at many AWNF steps.")
    lines.append("- If `boxed_like_*` categories are large, parser hardening may still help, but it is more likely to require robustness to malformed or repeated formatting than char-offset token alignment alone.")
    lines.append("- Persistent Bucket A counts within the same sample suggest a structural generation/parser mismatch rather than rare isolated misses.")

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote: {md_path}")
    print(f"Wrote: {json_path}")


if __name__ == "__main__":
    main()
