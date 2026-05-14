"""Phase 2: Offline AWNF decomposition for Math500.

Classifies `answer_window_not_found` (AWNF) events into:
  - Bucket A: span-aware parser fails (suspect parser output)
  - Bucket B: span-aware parser succeeds (true alignment failure — char-offset patch
    is the direct candidate)

Also reports Bucket C (window mismatch in valid events) using fast tokenizer +
offset_mapping to align the parser-reported (char_start, char_end) to token span,
then compare with the stored `window_start`/`window_end`.

Outputs both stdout summary and a markdown report under eval/analysis/.
"""

import argparse
import json
import os
import sys
from collections import Counter
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
DEFAULT_MODEL = (
    "/home/work/GFlowPO/jaeyoon/.cache/huggingface/hub/"
    "models--GSAI-ML--LLaDA-8B-Instruct/snapshots/"
    "08b83a6feb34df1a6011b80c3c00c7563e963b07"
)
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "analysis")


def char_span_to_token_span(offsets, char_start, char_end):
    """Return (token_start, token_end) — first/last token indices overlapping [char_start, char_end).

    Returns (None, None) if no token overlaps.
    """
    overlapping = [i for i, (s, e) in enumerate(offsets) if s < char_end and e > char_start]
    if not overlapping:
        return None, None
    return overlapping[0], overlapping[-1] + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=DEFAULT_RUN, help="Path to rank_0_generations.json")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to tokenizer model")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--task", default="math500")
    parser.add_argument("--skip-bucket-c", action="store_true", help="Skip Bucket C (no tokenizer)")
    args = parser.parse_args()

    print(f"Loading run: {args.run}")
    with open(args.run) as f:
        data = json.load(f)

    tokenizer = None
    if not args.skip_bucket_c:
        from transformers import AutoTokenizer

        print(f"Loading tokenizer: {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)

    skip_counter = Counter()
    total_events = 0

    awnf_total = 0
    awnf_bucket_a = 0
    awnf_bucket_b = 0
    awnf_bucket_b_examples = []
    awnf_bucket_a_examples = []

    valid_total = 0
    bucket_c_count = 0
    bucket_c_skipped_no_span = 0
    bucket_c_skipped_no_token = 0
    bucket_c_examples = []

    for s_idx, vd in enumerate(data["vote_debug"]):
        for step in vd["steps"]:
            total_events += 1
            sr = step.get("skip_reason")

            if sr is None:
                skip_counter["valid"] += 1
                valid_total += 1

                if not args.skip_bucket_c:
                    text = step.get("parsed_answer", "")
                    span_ans, span_cs, span_ce = parse_math_answer_with_span(text)
                    if span_cs < 0:
                        bucket_c_skipped_no_span += 1
                        continue

                    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
                    tok_start, tok_end = char_span_to_token_span(enc["offset_mapping"], span_cs, span_ce)
                    if tok_start is None:
                        bucket_c_skipped_no_token += 1
                        continue

                    win_s = step.get("window_start")
                    win_e = step.get("window_end")
                    if win_s is None or win_e is None:
                        continue

                    overlap_lo = max(tok_start, win_s)
                    overlap_hi = min(tok_end, win_e)
                    if overlap_hi <= overlap_lo:
                        bucket_c_count += 1
                        if len(bucket_c_examples) < 5:
                            bucket_c_examples.append({
                                "sample": s_idx,
                                "step": step["step"],
                                "span_ans": span_ans[:80] if isinstance(span_ans, str) else span_ans,
                                "span_tokens": (tok_start, tok_end),
                                "window": (win_s, win_e),
                            })
            else:
                skip_counter[sr] += 1
                if sr == "answer_window_not_found":
                    awnf_total += 1
                    text = step.get("parsed_answer", "")
                    span_ans, span_cs, span_ce = parse_math_answer_with_span(text)
                    if span_cs < 0:
                        awnf_bucket_a += 1
                        if len(awnf_bucket_a_examples) < 3:
                            awnf_bucket_a_examples.append({
                                "sample": s_idx,
                                "step": step["step"],
                                "tail": text[-120:],
                            })
                    else:
                        awnf_bucket_b += 1
                        if len(awnf_bucket_b_examples) < 5:
                            awnf_bucket_b_examples.append({
                                "sample": s_idx,
                                "step": step["step"],
                                "span_ans": span_ans[:80] if isinstance(span_ans, str) else span_ans,
                                "span_chars": (span_cs, span_ce),
                                "surface": text[span_cs:span_ce],
                            })

    bucket_b_over_awnf = awnf_bucket_b / awnf_total * 100 if awnf_total else 0
    bucket_b_over_total = awnf_bucket_b / total_events * 100 if total_events else 0
    bucket_c_pct = bucket_c_count / valid_total * 100 if valid_total else 0

    if bucket_b_over_awnf >= 50 and bucket_b_over_total >= 5:
        verdict = "GO — both thresholds met"
    elif bucket_b_over_awnf >= 50 or bucket_b_over_total >= 5:
        verdict = "PARTIAL — one of two thresholds met; review absolute numbers"
    else:
        verdict = "NO-GO heuristic — neither threshold met; reconsider Phase 4 priority"

    print()
    print("=" * 60)
    print(f"[{args.task} AWNF 분해]")
    print("=" * 60)
    print(f"Total events: {total_events}")
    for sr, c in skip_counter.most_common():
        print(f"  {sr}: {c} ({c/total_events*100:.1f}%)")
    print()
    print(f"Total AWNF events: {awnf_total}")
    print(f"  Bucket A (suspect parser output): {awnf_bucket_a} ({awnf_bucket_a/awnf_total*100:.1f}%)")
    print(f"  Bucket B (true alignment fail):   {awnf_bucket_b} ({bucket_b_over_awnf:.1f}%)")
    print()
    print(f"Bucket B / total_events: {bucket_b_over_total:.2f}%")
    print()
    print(f"Total valid events: {valid_total}")
    print(f"  Bucket C (window mismatch):       {bucket_c_count} ({bucket_c_pct:.2f}%)")
    if bucket_c_skipped_no_span or bucket_c_skipped_no_token:
        print(f"  (skipped from C analysis: no_span={bucket_c_skipped_no_span}, no_token={bucket_c_skipped_no_token})")
    print()
    print(f"Verdict: {verdict}")
    print()

    os.makedirs(args.out_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    out_path = os.path.join(args.out_dir, f"{args.task}_awnf_decomp_{date_str}.md")

    lines = []
    lines.append(f"# {args.task} AWNF Decomposition — {date_str}")
    lines.append("")
    lines.append(f"Run: `{args.run}`")
    lines.append("")
    lines.append("## Event distribution")
    lines.append("")
    lines.append(f"Total events: {total_events}")
    lines.append("")
    lines.append("| Category | Count | Share |")
    lines.append("|---|---:|---:|")
    for sr, c in skip_counter.most_common():
        lines.append(f"| {sr} | {c} | {c/total_events*100:.1f}% |")
    lines.append("")
    lines.append("## Bucket A/B — AWNF decomposition")
    lines.append("")
    lines.append(f"Total AWNF events: {awnf_total}")
    lines.append("")
    lines.append("| Bucket | Count | % of AWNF | % of total events |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| A (suspect parser output) | {awnf_bucket_a} | {awnf_bucket_a/awnf_total*100:.1f}% | "
        f"{awnf_bucket_a/total_events*100:.2f}% |"
    )
    lines.append(
        f"| B (true alignment fail)   | {awnf_bucket_b} | {bucket_b_over_awnf:.1f}% | {bucket_b_over_total:.2f}% |"
    )
    lines.append("")
    lines.append("## Bucket C — valid event window mismatch")
    lines.append("")
    lines.append(f"Total valid events: {valid_total}")
    lines.append(
        f"Bucket C (window/answer span no overlap): {bucket_c_count} ({bucket_c_pct:.2f}%)"
    )
    if bucket_c_skipped_no_span or bucket_c_skipped_no_token:
        lines.append(
            f"Skipped from C: no_span={bucket_c_skipped_no_span}, no_token={bucket_c_skipped_no_token}"
        )
    lines.append("")
    lines.append("## Go/No-Go gate")
    lines.append("")
    lines.append("| Threshold | Value | Met |")
    lines.append("|---|---:|:---:|")
    lines.append(f"| Bucket B / total_AWNF ≥ 50% | {bucket_b_over_awnf:.1f}% | "
                 f"{'✓' if bucket_b_over_awnf >= 50 else '✗'} |")
    lines.append(f"| Bucket B / total_events ≥ 5% | {bucket_b_over_total:.2f}% | "
                 f"{'✓' if bucket_b_over_total >= 5 else '✗'} |")
    lines.append("")
    lines.append(f"**Verdict**: {verdict}")
    lines.append("")
    if awnf_bucket_b_examples:
        lines.append("## Bucket B examples (parser succeeded, alignment failed)")
        lines.append("")
        for ex in awnf_bucket_b_examples:
            lines.append(f"- sample={ex['sample']}, step={ex['step']}")
            lines.append(f"    - span_ans: `{ex['span_ans']}`")
            lines.append(f"    - char span: {ex['span_chars']}")
            lines.append(f"    - surface: `{ex['surface']}`")
        lines.append("")
    if awnf_bucket_a_examples:
        lines.append("## Bucket A examples (parser failed)")
        lines.append("")
        for ex in awnf_bucket_a_examples:
            lines.append(f"- sample={ex['sample']}, step={ex['step']}")
            lines.append(f"    - tail of parsed_answer: `...{ex['tail']}`")
        lines.append("")
    if bucket_c_examples:
        lines.append("## Bucket C examples (valid event with off-target window)")
        lines.append("")
        for ex in bucket_c_examples:
            lines.append(f"- sample={ex['sample']}, step={ex['step']}")
            lines.append(f"    - span_ans: `{ex['span_ans']}`")
            lines.append(f"    - span tokens: {ex['span_tokens']}, stored window: {ex['window']}")
        lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
