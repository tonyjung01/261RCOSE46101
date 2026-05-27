import argparse
import json
import math
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np


def _rankdata(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def _pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return float("nan")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x, y):
    if len(x) < 2:
        return float("nan")
    return _pearson(_rankdata(x), _rankdata(y))


def _to_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_correct(parsed_answer, ground_truth):
    parsed_num = _to_float(parsed_answer)
    gt_num = _to_float(ground_truth)
    if parsed_num is not None and gt_num is not None:
        return math.isclose(parsed_num, gt_num, rel_tol=0.0, abs_tol=1e-9)
    return parsed_answer == ground_truth


def _load_events(run_path):
    payload = json.loads(Path(run_path).read_text())
    generations = payload["generations"]
    vote_debug = payload["vote_debug"]
    vote_method = payload.get("vote_method")

    events_by_sample = []
    total_step_records = 0
    valid_events = 0
    gap_events = 0

    for sample_idx, (gen, debug) in enumerate(zip(generations, vote_debug)):
        gt = gen["ground_truth"]
        sample_events = []
        for step_event in debug.get("steps", []):
            total_step_records += 1
            if step_event.get("skip_reason") is not None:
                continue
            if "raw_weight" not in step_event:
                continue
            valid_events += 1
            if "gap_values" in step_event:
                gap_events += 1
            sample_events.append(
                {
                    "sample_idx": sample_idx,
                    "step": int(step_event["step"]),
                    "total_steps": int(step_event["total_steps"]),
                    "parsed_answer": step_event.get("parsed_answer"),
                    "raw_weight": float(step_event["raw_weight"]),
                    "is_correct": _is_correct(step_event.get("parsed_answer"), gt),
                    "question": gen.get("question"),
                    "ground_truth": gt,
                }
            )
        events_by_sample.append(sample_events)

    return {
        "payload": payload,
        "events_by_sample": events_by_sample,
        "vote_method": vote_method,
        "total_step_records": total_step_records,
        "valid_events": valid_events,
        "gap_events": gap_events,
    }


def _flatten(events_by_sample):
    return [event for sample_events in events_by_sample for event in sample_events]


def _step_bin(step, total_steps):
    frac = (step + 1) / total_steps
    if frac <= 0.25:
        return "Q1 (0-25%)"
    if frac <= 0.50:
        return "Q2 (25-50%)"
    if frac <= 0.75:
        return "Q3 (50-75%)"
    return "Q4 (75-100%)"


def _collapse_contiguous(sample_events):
    if not sample_events:
        return []
    collapsed = []
    current = [sample_events[0]]
    for event in sample_events[1:]:
        if event["parsed_answer"] == current[-1]["parsed_answer"]:
            current.append(event)
        else:
            collapsed.append(_summarize_group(current))
            current = [event]
    collapsed.append(_summarize_group(current))
    return collapsed


def _collapse_unique(sample_events):
    grouped = defaultdict(list)
    for event in sample_events:
        grouped[event["parsed_answer"]].append(event)
    return [_summarize_group(group) for group in grouped.values()]


def _summarize_group(group):
    weights = [event["raw_weight"] for event in group]
    representative = dict(group[0])
    representative["raw_weight"] = float(np.median(weights))
    representative["step"] = int(np.median([event["step"] for event in group]))
    return representative


def _pairwise_winrate(events_by_sample, variant):
    numerator = 0
    denominator = 0

    for sample_events in events_by_sample:
        if variant == "raw":
            view = sample_events
        elif variant == "contiguous_dedup":
            view = _collapse_contiguous(sample_events)
        elif variant == "answer_level_unique":
            view = _collapse_unique(sample_events)
        else:
            raise ValueError(f"Unknown variant: {variant}")

        correct = [event for event in view if event["is_correct"]]
        wrong = [event for event in view if not event["is_correct"]]

        for correct_event in correct:
            for wrong_event in wrong:
                denominator += 1
                if correct_event["raw_weight"] > wrong_event["raw_weight"]:
                    numerator += 1

    return {
        "winrate": (numerator / denominator) if denominator else float("nan"),
        "numerator": numerator,
        "denominator": denominator,
    }


def analyze(run_path):
    loaded = _load_events(run_path)
    flat_events = _flatten(loaded["events_by_sample"])

    gaps = [event["raw_weight"] for event in flat_events]
    labels = [1.0 if event["is_correct"] else 0.0 for event in flat_events]

    unconditional = {
        "pearson": _pearson(gaps, labels),
        "spearman": _spearman(gaps, labels),
        "n_events": len(flat_events),
    }

    bins = defaultdict(list)
    for event in flat_events:
        bins[_step_bin(event["step"], event["total_steps"])].append(event)

    step_conditional = {}
    for bin_name in ["Q1 (0-25%)", "Q2 (25-50%)", "Q3 (50-75%)", "Q4 (75-100%)"]:
        events = bins.get(bin_name, [])
        step_conditional[bin_name] = {
            "pearson": _pearson(
                [event["raw_weight"] for event in events],
                [1.0 if event["is_correct"] else 0.0 for event in events],
            ),
            "spearman": _spearman(
                [event["raw_weight"] for event in events],
                [1.0 if event["is_correct"] else 0.0 for event in events],
            ),
            "n_events": len(events),
        }

    within_sample = {
        "raw": _pairwise_winrate(loaded["events_by_sample"], "raw"),
        "contiguous_dedup": _pairwise_winrate(loaded["events_by_sample"], "contiguous_dedup"),
        "answer_level_unique": _pairwise_winrate(loaded["events_by_sample"], "answer_level_unique"),
    }

    return {
        "run_path": str(run_path),
        "vote_method": loaded["vote_method"],
        "n_samples": len(loaded["events_by_sample"]),
        "total_step_records": loaded["total_step_records"],
        "valid_events": loaded["valid_events"],
        "gap_events": loaded["gap_events"],
        "all_valid_events_have_gap_values": loaded["valid_events"] == loaded["gap_events"],
        "unconditional": unconditional,
        "step_conditional": step_conditional,
        "within_sample": within_sample,
    }


def _fmt(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan"
    return f"{value:.4f}"


def to_markdown(result):
    lines = []
    lines.append("# GSM8K Gap-Correctness Analysis")
    lines.append("")
    lines.append(f"- Date: {date.today().isoformat()}")
    lines.append(f"- Run: `{result['run_path']}`")
    lines.append(f"- Stored vote method: `{result['vote_method']}`")
    lines.append(f"- Samples: `{result['n_samples']}`")
    lines.append(f"- Valid events: `{result['valid_events']}`")
    lines.append(
        f"- Gap field coverage on valid events: `{result['gap_events']}/{result['valid_events']}`"
        f" (`all_valid_events_have_gap_values={result['all_valid_events_have_gap_values']}`)"
    )
    lines.append("")
    lines.append("## Unconditional")
    lines.append("")
    lines.append(
        f"- Pearson: `{_fmt(result['unconditional']['pearson'])}`"
        f" over `{result['unconditional']['n_events']}` events"
    )
    lines.append(
        f"- Spearman: `{_fmt(result['unconditional']['spearman'])}`"
        f" over `{result['unconditional']['n_events']}` events"
    )
    lines.append("")
    lines.append("## Step-Conditional")
    lines.append("")
    for bin_name, stats in result["step_conditional"].items():
        lines.append(
            f"- {bin_name}: Pearson `{_fmt(stats['pearson'])}`,"
            f" Spearman `{_fmt(stats['spearman'])}`, n=`{stats['n_events']}`"
        )
    lines.append("")
    lines.append("## Within-Sample Pairwise Win-Rate")
    lines.append("")
    for key, label in [
        ("raw", "Raw"),
        ("contiguous_dedup", "Contiguous dedup"),
        ("answer_level_unique", "Answer-level unique"),
    ]:
        stats = result["within_sample"][key]
        lines.append(
            f"- {label}: `{_fmt(stats['winrate'])}`"
            f" (`{stats['numerator']}/{stats['denominator']}` correct>wrong pairs)"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- This analysis treats the stored per-step `raw_weight` as the gap-derived signal and uses"
        " the saved GSM8K ground truth for correctness."
    )
    lines.append(
        "- If the run directory name suggests `everpass` but the stored `vote_method` is a confidence-gap"
        " variant, interpret the result based on the stored method metadata rather than the directory name."
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    result = analyze(args.run_json)

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    output_md.write_text(to_markdown(result))

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nWrote: {output_json}")
    print(f"Wrote: {output_md}")


if __name__ == "__main__":
    main()
