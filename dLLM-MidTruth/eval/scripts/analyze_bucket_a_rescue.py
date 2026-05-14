"""Offline upper-bound analysis for Math500 Bucket A parser hardening.

This script does NOT change the official parser used in evaluation.
Instead, it asks a diagnostic question:

  If we applied a small set of *conservative formatting-only heuristics* to
  Bucket A AWNF events, how many of them would become parseable, and how much
  could that matter to sample-level `exp_only` voting?

The goal is to estimate whether parser-side hardening is likely to be
worthwhile before touching the main evaluation pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from scripts.analyze_bucket_a import CATEGORY_ORDER, classify_bucket_a  # noqa: E402
from utils.parsers import is_equiv, last_boxed_only_string, remove_boxed  # noqa: E402
from utils.span_parsers import parse_math_answer_with_span  # noqa: E402


DEFAULT_RUN = os.path.join(
    REPO_ROOT,
    "outputs/LLaDA-8B-Instruct/20260430_cgap_answer_window5_prob_mean_rawsum_bs4_all_debug/"
    "math500_gen128_steps64_vote_confidence_gap_answer_window5_prob_mean_rawsum_bs4/"
    "rank_0_generations.json",
)
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "analysis")
ALPHA = 5.0


def _exp_weight(step, total_steps):
    return math.exp(step / total_steps * ALPHA)


def _exp_vote(events):
    scores = {}
    for ev in events:
        ans = ev.get("parsed_answer")
        if ans is None:
            continue
        scores[ans] = scores.get(ans, 0.0) + ev["exp_weight"]
    if not scores:
        return None
    return max(scores.items(), key=lambda kv: kv[1])[0]


def _looks_mathish(text):
    text = text or ""
    return any(ch.isdigit() for ch in text) or ("\\frac" in text) or ("\\pi" in text) or ("sqrt" in text)


def _parse_math_answer_local(raw_generation):
    """Local mirror of utils.math500.parse_math_answer without dataset imports."""
    parsed_answer = None
    try:
        parsed_answer = remove_boxed(last_boxed_only_string(raw_generation))
    except Exception:
        parsed_answer = None
    if not parsed_answer:
        answer_match = re.search(r"<answer>(.*?)</answer>", raw_generation, re.DOTALL)
        if answer_match:
            parsed_answer = answer_match.group(1).strip()
    return parsed_answer


def _strip_soft(text):
    text = text.strip()
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" ,;:$")
    return text


def _extract_tail_before_closing_answer(text):
    if "</answer>" not in text:
        return None
    before = text.split("</answer>")[0]
    if "<answer>" in before:
        before = before.split("<answer>")[-1]
    lines = [ln.strip() for ln in before.splitlines() if ln.strip()]
    tail = lines[-1] if lines else before.strip()
    tail = _strip_soft(tail)
    if not _looks_mathish(tail):
        return None
    return tail


def _find_relaxed_boxed_content(text):
    lowered = text.lower()
    idx = lowered.rfind("boxed")
    if idx < 0:
        return None
    # Try to find the first opening brace reasonably close to the boxed marker.
    brace_idx = text.find("{", idx)
    if brace_idx < 0:
        # No brace at all: allow tail-to-closing-tag fallback after the marker.
        tail = text[idx + 5 :]
        tail = tail.split("</answer>")[0]
        tail = _strip_soft(tail)
        return tail if _looks_mathish(tail) else None

    depth = 1
    chars = []
    i = brace_idx + 1
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
            chars.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
            chars.append(ch)
        elif text.startswith("</answer>", i):
            # malformed boxed that runs into answer close tag
            break
        else:
            chars.append(ch)
        i += 1
    content = _strip_soft("".join(chars))
    return content if _looks_mathish(content) else None


def _normalize_boxed_repetition(text):
    # Conservative: only collapse repeated "boxed" tokens, do not alter math body.
    return re.sub(r"(\\?boxed){2,}", r"\\boxed", text, flags=re.IGNORECASE)


def _try_rescue(text):
    """Return dict with heuristic name and recovered answer, or None."""
    # H1: repeated/corrupted boxed marker collapse, then relaxed boxed parse.
    norm = _normalize_boxed_repetition(text)
    if norm != text:
        content = _find_relaxed_boxed_content(norm)
        if content:
            synthetic = f"<answer>\\boxed{{{content}}}</answer>"
            parsed = _parse_math_answer_local(synthetic)
            if parsed:
                return {"heuristic": "collapse_boxed_then_relaxed_boxed", "synthetic": synthetic, "parsed_answer": parsed}

    # H2: relaxed boxed extraction on original text.
    content = _find_relaxed_boxed_content(text)
    if content:
        synthetic = f"<answer>\\boxed{{{content}}}</answer>"
        parsed = _parse_math_answer_local(synthetic)
        if parsed:
            return {"heuristic": "relaxed_boxed", "synthetic": synthetic, "parsed_answer": parsed}

    # H3: closing-tag tail fallback if answer closing tag exists but span parse failed.
    tail = _extract_tail_before_closing_answer(text)
    if tail:
        synthetic = f"<answer>{tail}</answer>"
        parsed = _parse_math_answer_local(synthetic)
        if parsed:
            return {"heuristic": "closing_tag_tail", "synthetic": synthetic, "parsed_answer": parsed}

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    with open(args.run) as f:
        data = json.load(f)

    total_bucket_a = 0
    rescued_events = 0
    rescued_correct = 0
    bucket_category_counts = Counter()
    rescued_by_category = Counter()
    rescued_correct_by_category = Counter()
    rescued_by_heuristic = Counter()
    rescued_correct_by_heuristic = Counter()
    examples = defaultdict(list)

    sample_base = []
    sample_rescued = []

    for sample_idx, (gen, vd) in enumerate(zip(data["generations"], data["vote_debug"])):
        gt = gen["ground_truth"]
        base_valid = []
        rescued_valid = []

        for step in vd["steps"]:
            if step.get("skip_reason") is None:
                rw = step.get("raw_weight")
                pa = step.get("parsed_answer")
                if rw is not None and pa is not None:
                    ev = {
                        "step": step["step"],
                        "total_steps": step["total_steps"],
                        "parsed_answer": pa,
                        "exp_weight": _exp_weight(step["step"], step["total_steps"]),
                    }
                    base_valid.append(ev)
                    rescued_valid.append(ev)
                continue

            if step.get("skip_reason") != "answer_window_not_found":
                continue

            text = step.get("parsed_answer", "")
            _, char_start, _ = parse_math_answer_with_span(text)
            if char_start >= 0:
                continue

            total_bucket_a += 1
            category = classify_bucket_a(text)
            bucket_category_counts[category] += 1

            rescue = _try_rescue(text)
            if rescue is None:
                continue

            rescued_events += 1
            rescued_by_category[category] += 1
            rescued_by_heuristic[rescue["heuristic"]] += 1
            is_corr = bool(is_equiv(rescue["parsed_answer"], gt))
            if is_corr:
                rescued_correct += 1
                rescued_correct_by_category[category] += 1
                rescued_correct_by_heuristic[rescue["heuristic"]] += 1

            rescued_valid.append(
                {
                    "step": step["step"],
                    "total_steps": step["total_steps"],
                    "parsed_answer": rescue["parsed_answer"],
                    "exp_weight": _exp_weight(step["step"], step["total_steps"]),
                }
            )

            if len(examples[rescue["heuristic"]]) < 3:
                examples[rescue["heuristic"]].append(
                    {
                        "sample": sample_idx,
                        "step": step["step"],
                        "category": category,
                        "text_snippet": text[:220].replace("\n", "\\n"),
                        "synthetic": rescue["synthetic"][:220].replace("\n", "\\n"),
                        "parsed_answer": rescue["parsed_answer"],
                        "is_correct_vs_gt": is_corr,
                    }
                )

        base_vote = _exp_vote(base_valid)
        rescued_vote = _exp_vote(rescued_valid)
        base_ok = bool(is_equiv(base_vote, gt))
        rescued_ok = bool(is_equiv(rescued_vote, gt))
        sample_base.append(base_ok)
        sample_rescued.append(rescued_ok)

    base_acc = sum(sample_base) / len(sample_base)
    rescued_acc = sum(sample_rescued) / len(sample_rescued)
    changed = 0
    fixed = 0
    hurt = 0
    for b, r in zip(sample_base, sample_rescued):
        if b != r:
            changed += 1
            if (not b) and r:
                fixed += 1
            if b and (not r):
                hurt += 1

    report = {
        "run": args.run,
        "total_bucket_a": total_bucket_a,
        "rescued_events": rescued_events,
        "rescued_event_rate": (rescued_events / total_bucket_a) if total_bucket_a else 0.0,
        "rescued_correct_events": rescued_correct,
        "rescued_correct_event_rate": (rescued_correct / total_bucket_a) if total_bucket_a else 0.0,
        "bucket_category_counts": dict(bucket_category_counts),
        "rescued_by_category": dict(rescued_by_category),
        "rescued_correct_by_category": dict(rescued_correct_by_category),
        "rescued_by_heuristic": dict(rescued_by_heuristic),
        "rescued_correct_by_heuristic": dict(rescued_correct_by_heuristic),
        "sample_level": {
            "n_samples": len(sample_base),
            "base_exp_acc": base_acc,
            "rescued_exp_acc": rescued_acc,
            "delta": rescued_acc - base_acc,
            "changed_samples": changed,
            "fixed_samples": fixed,
            "hurt_samples": hurt,
        },
        "examples": dict(examples),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    json_path = os.path.join(args.out_dir, f"math500_bucket_a_rescue_{date_str}.json")
    md_path = os.path.join(args.out_dir, f"math500_bucket_a_rescue_{date_str}.md")

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append(f"# Math500 Bucket A Rescue Upper Bound — {date_str}")
    lines.append("")
    lines.append("This is an offline diagnostic only. It does **not** change the official parser or any reported mainline metric.")
    lines.append("")
    lines.append(f"Run: `{args.run}`")
    lines.append("")
    lines.append("## Event-level rescue")
    lines.append("")
    lines.append(f"- Total Bucket A events: `{total_bucket_a}`")
    lines.append(f"- Rescued (parseable) events: `{rescued_events}` (`{(rescued_events / total_bucket_a * 100) if total_bucket_a else 0:.1f}%`)")
    lines.append(f"- Rescued events matching ground truth: `{rescued_correct}` (`{(rescued_correct / total_bucket_a * 100) if total_bucket_a else 0:.1f}%`)")
    lines.append("")
    lines.append("| Category | Bucket A count | Rescued | Rescued % | GT-equivalent rescued | GT-equivalent % |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for cat in CATEGORY_ORDER:
        total = bucket_category_counts[cat]
        rec = rescued_by_category[cat]
        rec_ok = rescued_correct_by_category[cat]
        lines.append(
            f"| {cat} | {total} | {rec} | {(rec / total * 100) if total else 0:.1f}% | {rec_ok} | {(rec_ok / total * 100) if total else 0:.1f}% |"
        )
    lines.append("")
    lines.append("| Heuristic | Rescued | Rescued % of Bucket A | GT-equivalent rescued |")
    lines.append("|---|---:|---:|---:|")
    for name, count in rescued_by_heuristic.most_common():
        lines.append(
            f"| {name} | {count} | {(count / total_bucket_a * 100) if total_bucket_a else 0:.1f}% | {rescued_correct_by_heuristic[name]} |"
        )
    lines.append("")
    lines.append("## Sample-level `exp_only` upper bound")
    lines.append("")
    lines.append(f"- Base `exp_only` acc: `{base_acc:.2%}`")
    lines.append(f"- Rescued `exp_only` acc: `{rescued_acc:.2%}`")
    lines.append(f"- Delta: `{rescued_acc - base_acc:+.2%}`")
    lines.append(f"- Changed samples: `{changed}`")
    lines.append(f"- Fixed samples: `{fixed}`")
    lines.append(f"- Hurt samples: `{hurt}`")
    lines.append("")
    lines.append("## Example recoveries")
    lines.append("")
    for hname, exs in examples.items():
        lines.append(f"### {hname}")
        lines.append("")
        for ex in exs:
            lines.append(f"- sample `{ex['sample']}`, step `{ex['step']}`, category `{ex['category']}`, GT-match `{ex['is_correct_vs_gt']}`")
            lines.append(f"  - raw: `{ex['text_snippet']}`")
            lines.append(f"  - synthetic: `{ex['synthetic']}`")
            lines.append(f"  - parsed: `{ex['parsed_answer']}`")
        lines.append("")
    lines.append("## Current read")
    lines.append("")
    lines.append("- If rescue rate is low or sample-level delta is negligible, that suggests parser-side hardening is unlikely to be a first-order fix for the current Math500 bottleneck.")
    lines.append("- If rescueable events concentrate in `boxed_like_*` / `boxedboxed_*`, then small formatting heuristics may be worth documenting as future work even if they stay outside the mainline parser.")
    lines.append("- Because this analysis preserves the official parser/evaluator and only asks for an offline upper bound, it should be read as a bottleneck diagnosis rather than a new benchmark result.")

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote: {md_path}")
    print(f"Wrote: {json_path}")


if __name__ == "__main__":
    main()
