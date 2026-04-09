import argparse
import json
import os
import re
from glob import glob

from utils.countdown import parse_ctd_answer
from utils.gsm8k import parse_gsm_answer
from utils.math500 import parse_math_answer
from utils.parsers import is_equiv
from utils.svamp import parse_svamp_answer


GENERATION_FILE_RE = re.compile(r"rank_\d+_generations\.json$")
DIR_META_RE = re.compile(
    r"^(?P<dataset>[^_]+)_gen_(?P<gen_length>\d+)_steps_(?P<diffusion_steps>\d+)_temp_(?P<temperature>[^_]+)(?P<suffix>.*)$"
)


def parse_answer(dataset, text):
    if "svamp" in dataset:
        return parse_svamp_answer(text)
    if "gsm" in dataset:
        return parse_gsm_answer(text)
    if "math" in dataset:
        return parse_math_answer(text)
    if "countdown" in dataset:
        return parse_ctd_answer(text)
    raise NotImplementedError(f"Dataset {dataset} is not supported for answer parsing.")


def eval_result(dataset, answer, ground_truth):
    def validate_equation(equation_str, available_numbers):
        try:
            numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]
            available_numbers = sorted(available_numbers)
            numbers_in_eq = sorted(numbers_in_eq)
            return numbers_in_eq == available_numbers
        except Exception:
            return False

    def evaluate_equation(equation_str):
        try:
            allowed_pattern = r"^[\d+\-*/().\s]+$"
            if not re.match(allowed_pattern, equation_str):
                raise ValueError("Invalid characters in equation.")
            return eval(equation_str.strip(), {"__builtins__": None}, {})
        except Exception:
            return float("inf")

    if "svamp" in dataset or "gsm" in dataset:
        if answer is None:
            return False
        return float(answer) == float(ground_truth)
    if "math" in dataset:
        if answer is None:
            return False
        return is_equiv(answer, ground_truth)
    if "countdown" in dataset:
        is_valid = validate_equation(answer, ground_truth[0])
        if is_valid:
            result = evaluate_equation(answer)
            if abs(result - ground_truth[1]) < 1e-5:
                return True
    return False


def is_generation_file(filename):
    return GENERATION_FILE_RE.match(filename) is not None


def find_eval_directories(path):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")

    if os.path.isfile(path):
        raise ValueError(f"Expected a directory, got file: {path}")

    direct_files = [name for name in os.listdir(path) if is_generation_file(name)]
    if direct_files:
        return [path]

    directories = []
    for root, _, files in os.walk(path):
        if any(is_generation_file(filename) for filename in files):
            directories.append(root)

    directories.sort()
    if not directories:
        raise FileNotFoundError(f"No generation JSON files found under: {path}")
    return directories


def infer_metadata_from_directory(directory):
    directory = os.path.abspath(directory)
    folder = os.path.basename(directory)
    parent = os.path.basename(os.path.dirname(directory))
    grandparent = os.path.basename(os.path.dirname(os.path.dirname(directory)))

    if grandparent == "outputs":
        model_name = parent
        run_name = None
    else:
        model_name = grandparent
        run_name = parent

    metadata = {
        "directory": directory,
        "folder": folder,
        "model_name": model_name,
        "run_name": run_name,
    }

    match = DIR_META_RE.match(folder)
    if not match:
        return metadata

    metadata.update(
        {
            "dataset": match.group("dataset"),
            "gen_length": int(match.group("gen_length")),
            "diffusion_steps": int(match.group("diffusion_steps")),
            "temperature": match.group("temperature"),
        }
    )

    suffix = match.group("suffix")
    metadata["enable_vote"] = "vote" in suffix

    vote_match = re.search(r"vote(?:_(?P<vote_method>[a-z]+))?(?:_a(?P<alpha>[^_]+))?", suffix)
    if vote_match:
        if vote_match.group("vote_method"):
            metadata["vote_method"] = vote_match.group("vote_method")
        if vote_match.group("alpha"):
            metadata["alpha"] = vote_match.group("alpha")

    batch_match = re.search(r"_bs(?P<batch_size>\d+)", suffix)
    if batch_match:
        metadata["batch_size"] = int(batch_match.group("batch_size"))

    return metadata


def _first_non_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def summarize_directory(directory):
    json_files = sorted(glob(os.path.join(directory, "*.json")))
    json_files = [path for path in json_files if is_generation_file(os.path.basename(path))]
    if not json_files:
        raise FileNotFoundError(f"No generation JSON files found in directory: {directory}")

    inferred = infer_metadata_from_directory(directory)

    final_correct_questions = 0
    vote_correct_questions = 0
    total_questions = 0
    metadata = dict(inferred)

    for index, json_file in enumerate(json_files):
        with open(json_file, "r") as handle:
            data = json.load(handle)

        if index == 0:
            metadata.update(
                {
                    "dataset": _first_non_none(data.get("dataset"), metadata.get("dataset")),
                    "model_name": _first_non_none(data.get("model_name"), metadata.get("model_name")),
                    "gen_length": _first_non_none(data.get("gen_length"), metadata.get("gen_length")),
                    "diffusion_steps": _first_non_none(data.get("diffusion_steps"), metadata.get("diffusion_steps")),
                    "temperature": _first_non_none(data.get("temperature"), metadata.get("temperature")),
                    "batch_size": _first_non_none(data.get("batch_size"), metadata.get("batch_size")),
                    "enable_vote": _first_non_none(data.get("enable_vote"), metadata.get("enable_vote")),
                    "vote_method": _first_non_none(data.get("vote_method"), metadata.get("vote_method")),
                    "alpha": _first_non_none(data.get("alpha"), metadata.get("alpha")),
                    "model_path": data.get("model_path"),
                    "checkpoint_path": data.get("checkpoint_path"),
                }
            )

        generations = data["generations"]
        vote_enabled = metadata.get("enable_vote")
        if vote_enabled is None:
            vote_enabled = any("vote_answer" in generation for generation in generations)
            metadata["enable_vote"] = vote_enabled

        if vote_enabled:
            for generation in generations:
                vote_answer = generation.get("vote_answer")
                final_answer = generation.get("final_answer")
                ground_truth = generation["ground_truth"]
                vote_correct_questions += eval_result(metadata["dataset"], vote_answer, ground_truth)
                final_correct_questions += eval_result(metadata["dataset"], final_answer, ground_truth)
                total_questions += 1
        else:
            for generation in generations:
                generated_text = generation["generations"]
                final_answer = parse_answer(metadata["dataset"], generated_text)
                ground_truth = generation["ground_truth"]
                final_correct_questions += eval_result(metadata["dataset"], final_answer, ground_truth)
                total_questions += 1

    result = {
        "directory": directory,
        "dataset": metadata.get("dataset"),
        "model_name": metadata.get("model_name"),
        "run_name": metadata.get("run_name"),
        "gen_length": metadata.get("gen_length"),
        "diffusion_steps": metadata.get("diffusion_steps"),
        "temperature": metadata.get("temperature"),
        "batch_size": metadata.get("batch_size"),
        "enable_vote": bool(metadata.get("enable_vote")),
        "vote_method": metadata.get("vote_method"),
        "alpha": metadata.get("alpha"),
        "model_path": metadata.get("model_path"),
        "checkpoint_path": metadata.get("checkpoint_path"),
        "num_files": len(json_files),
        "total_questions": total_questions,
        "final_accuracy": final_correct_questions / total_questions * 100,
    }
    if result["enable_vote"]:
        result["vote_accuracy"] = vote_correct_questions / total_questions * 100
    return result


def print_result(result):
    print(f"directory: {result['directory']}")
    print(
        "config: "
        f"dataset={result.get('dataset')} "
        f"model={result.get('model_name')} "
        f"run={result.get('run_name')} "
        f"gen_length={result.get('gen_length')} "
        f"steps={result.get('diffusion_steps')} "
        f"temp={result.get('temperature')} "
        f"batch_size={result.get('batch_size')} "
        f"vote={result.get('enable_vote')} "
        f"vote_method={result.get('vote_method')} "
        f"alpha={result.get('alpha')}"
    )
    print(f"total process questions: {result['total_questions']}")
    print(f"Final answer accuracy: {result['final_accuracy']:.2f}%")
    if result.get("enable_vote"):
        print(f"Vote answer accuracy: {result['vote_accuracy']:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory",
        nargs="+",
        help="One or more output directories. Each path can be a single eval directory or a run root.",
    )
    args = parser.parse_args()

    all_results = []
    for directory in args.directory:
        for eval_directory in find_eval_directories(directory):
            all_results.append(summarize_directory(eval_directory))

    for index, result in enumerate(all_results):
        if index > 0:
            print()
        print_result(result)


if __name__ == "__main__":
    main()
