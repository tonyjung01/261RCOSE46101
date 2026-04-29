import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from peft import PeftModel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from generate import generate, parse_vote_method_config
from utils.countdown import CTDDataset, parse_ctd_answer
from utils.gsm8k import GSM8KDataset, parse_gsm_answer
from utils.math500 import MATH500Dataset, parse_math_answer
from utils.svamp import SVAMPDataset, parse_svamp_answer

DATASET_MAP = {
    "gsm8k": GSM8KDataset,
    "math": MATH500Dataset,
    "countdown": CTDDataset,
    "svamp": SVAMPDataset,
}

PARSE_MAP = {
    "gsm8k": parse_gsm_answer,
    "math": parse_math_answer,
    "countdown": parse_ctd_answer,
    "svamp": parse_svamp_answer,
}


def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_ddp():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def _build_vote_summary(sample_vote_debug):
    return {
        "valid_events": sample_vote_debug["valid_events"],
        "skipped_events": sample_vote_debug["skipped_events"],
        "skip_counts": sample_vote_debug["skip_counts"],
        "top_scores": sample_vote_debug["final_scores"][:3],
        "vote_method": sample_vote_debug["vote_method"],
        "vote_skip_first_ratio": sample_vote_debug["vote_config"].get("skip_first_ratio", 0.0),
        "vote_start_step": sample_vote_debug["vote_config"].get("start_step", 0),
    }


def evaluate(
    model,
    tokenizer,
    dataloader,
    gen_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    steps=64,
    block_length=32,
    enable_vote=False,
    parse_answer_func=None,
    vote_method=None,
    alpha=None,
    save_vote_debug=False,
    vote_skip_first_ratio=0.0,
):
    model.eval()
    total_processed = torch.tensor(0, device=model.device)
    wall_times = []
    all_generations = []
    all_vote_debug = []
    device = model.device

    for batch in tqdm(dataloader, disable=(dist.get_rank() != 0)):
        start_time = time.time()
        input_ids = batch["input_ids"].to(device)
        gt_answers = batch["answers"]
        questions = batch["questions"]
        prompts = batch["prompts"]

        out, vote_answers, vote_debug = generate(
            model,
            input_ids,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking="low_confidence",
            enable_vote=enable_vote,
            tokenizer=tokenizer,
            parse_answer_func=parse_answer_func,
            vote_method=vote_method,
            alpha=alpha,
            save_vote_debug=save_vote_debug,
            vote_skip_first_ratio=vote_skip_first_ratio,
        )

        generated_texts = tokenizer.batch_decode(out[:, -gen_length:], skip_special_tokens=False)

        if enable_vote:
            example_result = [
                {
                    "question": questions[j],
                    "prompt_input": prompts[j],
                    "generations": generated_texts[j],
                    "vote_answer": vote_answers[j],
                    "final_answer": parse_answer_func(generated_texts[j]),
                    "ground_truth": gt_answers[j],
                    "vote_summary": _build_vote_summary(vote_debug[j]),
                }
                for j in range(len(gt_answers))
            ]
            if save_vote_debug:
                all_vote_debug.extend(vote_debug)
        else:
            example_result = [
                {
                    "question": questions[j],
                    "prompt_input": prompts[j],
                    "generations": generated_texts[j],
                    "ground_truth": gt_answers[j],
                }
                for j in range(len(gt_answers))
            ]

        all_generations.extend(example_result)
        total_processed += len(generated_texts)
        wall_times.append(time.time() - start_time)

        if dist.get_rank() == 0:
            idx = random.randint(0, len(questions) - 1)
            print(f"Question: {questions[idx]}")
            print("-" * 50)
            print("Generation:")
            print(generated_texts[idx])
            print("-" * 50)
            print(f"Ground truth: {gt_answers[idx]}")
            if enable_vote:
                print("-" * 50)
                print(f"Vote answer: {vote_answers[idx]}")
                print(f"Vote summary: {_build_vote_summary(vote_debug[idx])}")

    avg_wall_time = sum(wall_times) / len(wall_times)
    metrics = {
        "wall_time": avg_wall_time,
        "generations": all_generations,
        "total_processed": total_processed.item(),
        "vote_debug": all_vote_debug if enable_vote and save_vote_debug else None,
    }
    return metrics


class CustomDistributedSampler(DistributedSampler):
    """
    From torch docs:
    drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas

    We want drop_last = False, but don't want to have extra padding indices. Hence using a custom sampler.
    """

    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        drop_last=False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil((len(self.dataset) - self.num_replicas) / self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.total_size = len(self.dataset)
            self.num_samples = len(self.dataset) // self.num_replicas + int(rank < (self.total_size % self.num_replicas))

        self.shuffle = shuffle
        self.seed = seed


if __name__ == "__main__":
    init_seed(42)

    local_rank = setup_ddp()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/data1/shared/LLaDA-8B-Instruct/")
    parser.add_argument("--model_name", type=str, default="LLaDA-8B-Instruct")
    parser.add_argument("--few_shot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--dataset", type=str, choices=["gsm8k", "math", "countdown", "svamp", "human_eval_plus"], default="gsm8k"
    )
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--diffusion_steps", type=int, default=64)
    parser.add_argument("--add_reasoning", action="store_true")
    parser.add_argument("--dont_save", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results/")
    parser.add_argument("--dont_use_box", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for generation.")
    parser.add_argument(
        "--enable_vote",
        action="store_true",
        help="Whether to enable vote functionality.",
    )
    parser.add_argument(
        "--vote_method",
        type=str,
        default=None,
        help=(
            "Voting method to use. Supported values: fixed, linear, exp, or "
            "confidence_gap_<region>_<window>_<reduce>_<scale>."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Alpha parameter used for 'exp' voting method.",
    )
    parser.add_argument(
        "--save_vote_debug",
        action="store_true",
        help="Save per-step vote debug traces for analysis.",
    )
    parser.add_argument(
        "--vote_skip_first_ratio",
        type=float,
        default=0.0,
        help="Skip the first ratio of diffusion steps when accumulating vote weights.",
    )

    args = parser.parse_args()

    num_evals = {"gsm8k": -1, "math": -1, "svamp": -1, "countdown": 256}

    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(
        local_rank
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if args.checkpoint_path:
        model = PeftModel.from_pretrained(model, args.checkpoint_path, torch_dtype=torch.bfloat16).to(local_rank)

        if dist.get_world_size() > 1:
            dist.barrier()
            for param in model.parameters():
                dist.broadcast(param.data, src=0)
            print(f"Rank {local_rank}: Parameters synchronized")

    dataset = DATASET_MAP[args.dataset](
        tokenizer,
        subsample=num_evals[args.dataset],
        num_examples=args.few_shot,
        add_reasoning=True,
    )

    parse_func = PARSE_MAP.get(args.dataset, None) if args.enable_vote else None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=CustomDistributedSampler(dataset, shuffle=False),
        collate_fn=dataset.collate_fn,
    )

    if len(args.checkpoint_path):
        model_name = args.checkpoint_path.split("/")
        model_name = model_name[-2] + "_" + model_name[-1]
    else:
        model_name = args.model_name

    if args.few_shot > 0:
        model_name = model_name + f"_fs{args.few_shot}"

    if len(args.suffix) > 0:
        model_name = model_name + f"_{args.suffix}"

    vote_method_details = None
    if args.enable_vote and args.vote_method is not None:
        vote_method_details = parse_vote_method_config(
            args.vote_method,
            alpha=args.alpha,
            skip_first_ratio=args.vote_skip_first_ratio,
        )
        vote_method_details["start_step"] = math.ceil(args.diffusion_steps * args.vote_skip_first_ratio)

    os.makedirs(args.output_dir, exist_ok=True)
    filename = f"{args.output_dir}/rank_{dist.get_rank()}_generations.json"
    print(f"Saving generations to {filename}")

    metrics = evaluate(
        model,
        tokenizer,
        dataloader,
        gen_length=args.gen_length,
        block_length=args.block_length,
        steps=args.diffusion_steps,
        temperature=args.temperature,
        enable_vote=args.enable_vote,
        parse_answer_func=parse_func,
        vote_method=args.vote_method,
        alpha=args.alpha,
        save_vote_debug=args.save_vote_debug,
        vote_skip_first_ratio=args.vote_skip_first_ratio,
    )

    if not args.dont_save:
        payload = {
            "generations": metrics["generations"],
            "metrics": {
                "wall_time": metrics["wall_time"],
                "total_processed": metrics["total_processed"],
            },
            "dataset": args.dataset,
            "model_name": model_name,
            "model_path": args.model_path,
            "checkpoint_path": args.checkpoint_path,
            "batch_size": args.batch_size,
            "gen_length": args.gen_length,
            "diffusion_steps": args.diffusion_steps,
            "block_length": args.block_length,
            "temperature": args.temperature,
            "enable_vote": args.enable_vote,
            "vote_method": args.vote_method,
            "vote_method_details": vote_method_details,
            "alpha": args.alpha,
            "save_vote_debug": args.save_vote_debug,
            "vote_skip_first_ratio": args.vote_skip_first_ratio,
        }
        if metrics["vote_debug"] is not None:
            payload["vote_debug"] = metrics["vote_debug"]

        with open(filename, "w") as f:
            json.dump(payload, f, indent=2)

    cleanup_ddp()
