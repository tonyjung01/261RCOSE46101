import argparse
import os
import re
from datetime import datetime


CONFIG_RE = re.compile(r"([a-zA-Z_]+)=([^ ]+)")
PERCENT_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)%")


def _parse_result_file(path):
    with open(path, "r") as handle:
        lines = [line.rstrip("\n") for line in handle]

    results = []
    current = {}

    def flush():
        nonlocal current
        if current:
            results.append(current)
            current = {}

    for line in lines:
        if not line.strip():
            flush()
            continue

        if line.startswith("directory: "):
            if current.get("directory"):
                flush()
            current["directory"] = line.split("directory: ", 1)[1].strip()
            continue

        if line.startswith("config: "):
            config_text = line.split("config: ", 1)[1]
            current.update(dict(CONFIG_RE.findall(config_text)))
            continue

        if line.startswith("total process questions: "):
            current["total_questions"] = int(line.split(": ", 1)[1])
            continue

        if line.startswith("Final answer accuracy: "):
            match = PERCENT_RE.search(line)
            if match:
                current["final_accuracy"] = float(match.group(1))
            continue

        if line.startswith("Vote answer accuracy: "):
            match = PERCENT_RE.search(line)
            if match:
                current["vote_accuracy"] = float(match.group(1))
            continue

    flush()
    return results


def _find_result_files(path):
    path = os.path.abspath(path)
    if os.path.isfile(path):
        return [path]

    candidates = []
    for filename in ("summary.txt", "accuracy.txt"):
        candidate = os.path.join(path, filename)
        if os.path.exists(candidate):
            candidates.append(candidate)

    if candidates:
        return candidates

    discovered = []
    for root, _, files in os.walk(path):
        if "summary.txt" in files:
            discovered.append(os.path.join(root, "summary.txt"))
        elif "accuracy.txt" in files:
            discovered.append(os.path.join(root, "accuracy.txt"))
    discovered.sort()
    return discovered


def _collect_results(paths):
    all_results = []
    for path in paths:
        files = _find_result_files(path)
        if not files:
            raise FileNotFoundError(f"No summary.txt or accuracy.txt found under: {path}")
        for file_path in files:
            all_results.extend(_parse_result_file(file_path))
    return all_results


def _format_markdown_table(results):
    header = [
        "| Dataset | Batch | Gen | Steps | Vote Method | Final Acc | Vote Acc | Directory |",
        "|---|---:|---:|---:|---|---:|---:|---|",
    ]
    rows = []
    for result in results:
        rows.append(
            "| {dataset} | {batch_size} | {gen_length} | {steps} | {vote_method} | {final:.2f} | {vote} | {directory} |".format(
                dataset=result.get("dataset", "-"),
                batch_size=result.get("batch_size", "-"),
                gen_length=result.get("gen_length", "-"),
                steps=result.get("steps", result.get("diffusion_steps", "-")),
                vote_method=result.get("vote_method", "none"),
                final=result.get("final_accuracy", 0.0),
                vote=(
                    f"{result['vote_accuracy']:.2f}"
                    if result.get("vote_accuracy") is not None
                    else "-"
                ),
                directory=result.get("directory", "-"),
            )
        )
    return "\n".join(header + rows)


def append_markdown_log(log_file, section_title, source_paths, results, note=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as handle:
        handle.write("\n\n")
        handle.write(f"## {timestamp} - {section_title}\n\n")
        handle.write("Source paths:\n")
        for path in source_paths:
            handle.write(f"- `{os.path.abspath(path)}`\n")
        handle.write("\n")
        if note:
            handle.write(f"Note: {note}\n\n")
        handle.write(_format_markdown_table(results))
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="Run roots, eval directories, or summary/accuracy files.")
    parser.add_argument("--log-file", required=True, help="Markdown file to append results to.")
    parser.add_argument("--section-title", default="Experiment Run", help="Title for the appended section.")
    parser.add_argument("--note", default="", help="Optional note to store with the appended section.")
    args = parser.parse_args()

    results = _collect_results(args.paths)
    append_markdown_log(
        log_file=os.path.abspath(args.log_file),
        section_title=args.section_title,
        source_paths=args.paths,
        results=results,
        note=args.note or None,
    )
    print(f"Appended {len(results)} result entries to {os.path.abspath(args.log_file)}")


if __name__ == "__main__":
    main()
