import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import torch.distributed as dist

def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    Using float16 for better performance while maintaining reasonable quality.
    """
    if temperature == 0.0:
        return logits  # Skip noise when temperature is 0

    # Use float32 instead of float64 for better performance
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits, dtype=torch.float32)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """
    Precompute the number of tokens to transition at each step.
    Optimized to be more efficient.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps

    # Create tensor once and modify in-place
    num_transfer_tokens = base.expand(-1, steps).clone()

    # Handle remainder more efficiently
    if remainder.sum() > 0:
        indices = torch.arange(steps, device=mask_index.device)
        mask = indices.unsqueeze(0) < remainder
        num_transfer_tokens[mask] += 1

    return num_transfer_tokens.to(torch.int64)

@torch.no_grad()
def generate(
    model,
    prompt,
    steps=64,
    gen_length=128,
    block_length=32,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    # === vote related parameter ===
    enable_vote=False,          # whether to enable voting mechanism
    tokenizer=None,             # tokenizer for parsing answer
    parse_answer_func=None,     # method to parse answer
    vote_method=None,           # ['fixed', 'linear', 'exp']
    alpha=None                  # hyperparameter for exp method
):
    """
    Optimized version of the generate function.
    """
    if enable_vote:
        if parse_answer_func is None or not callable(parse_answer_func):
            raise ValueError("When enable_vote=True, parse_answer_func must be provided and callable.")
        if tokenizer is None:
            raise ValueError("When enable_vote=True, tokenizer must be provided.")

        valid_methods = {"fixed", "linear", "exp"}
        if vote_method not in valid_methods:
            raise ValueError(f"When enable_vote=True, vote_method must be one of {valid_methods}.")

        if vote_method == "exp" and alpha is None:
            raise ValueError("When vote_method='exp', alpha parameter must be provided.")
        answer_count = [ {} for _ in range(prompt.shape[0]) ]
    else:
        tokenizer = None
        parse_answer_func = None
        vote_method = None
        alpha = None

    # Use mixed precision for faster computation
    with torch.autocast(device_type="cuda"):
        x = torch.full(
            (prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=prompt.device
        )
        x[:, : prompt.shape[1]] = prompt.clone()

        prompt_index = x != mask_id

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        steps_per_block = max(1, steps // num_blocks)
        for num_block in tqdm(range(num_blocks), disable=(dist.get_rank() != 0)):
            start_idx = prompt.shape[1] + num_block * block_length
            end_idx = prompt.shape[1] + (num_block + 1) * block_length

            block_mask_index = x[:, start_idx:end_idx] == mask_id
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

            for i in range(steps_per_block):
                step = i + num_block * steps_per_block
                mask_index = x == mask_id

                # Handle classifier-free guidance more efficiently
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)

                    # Get logits in a single forward pass
                    logits = model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(x).logits

                # Apply Gumbel noise for sampling
                logits_with_noise = add_gumbel_noise(logits, temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                # Handle remasking strategy
                if remasking == "low_confidence":
                    # Use float32 instead of float64 for better performance
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                elif remasking == "random":
                    x0_p = torch.rand(x0.shape, device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                # Ensure we don't process tokens beyond the current block
                x0_p[:, end_idx:] = -np.inf

                # Update masked tokens
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x0.device))

                # parse answer if voting is enabled
                if enable_vote:
                    import threading
                    lock = threading.Lock()
                    def vote_worker(x0_, step_, steps_, answer_count_, vote_method_, alpha_):
                        generated_texts = tokenizer.batch_decode(x0_[:, prompt.shape[1]:], skip_special_tokens=True)
                        for j in range(len(generated_texts)):
                            generated_text = generated_texts[j]
                            parsed_answer = parse_answer_func(generated_text)
                            if parsed_answer is not None:
                                with lock:
                                    if vote_method_ == "fixed":
                                        answer_count_[j][parsed_answer] = answer_count_[j].get(parsed_answer, 0) + 1
                                    elif vote_method_ == "linear":
                                        answer_count_[j][parsed_answer] = answer_count_[j].get(parsed_answer, 0) + (step_ + 1) / steps_
                                    elif vote_method_ == "exp":
                                        answer_count_[j][parsed_answer] = answer_count_[j].get(parsed_answer, 0) + np.exp(step_ / steps_ * alpha_)
                                    else:
                                        raise ValueError(f"Invalid vote_method: {vote_method_}")

                    if 'vote_threads' not in locals():
                        vote_threads = []
                    t = threading.Thread(target=vote_worker, args=(x0.clone(), step, steps, answer_count, vote_method, alpha))
                    t.start()
                    if 'vote_threads' in locals():
                        vote_threads.append(t)

                # Select tokens to transfer based on confidence
                for j in range(confidence.shape[0]):
                    num_tokens = num_transfer_tokens[j, i].item()
                    if num_tokens > 0:
                        _, select_indices = torch.topk(confidence[j], k=num_tokens)
                        x[j, select_indices] = x0[j, select_indices]

        if enable_vote:
            if 'vote_threads' in locals():
                for t in vote_threads:
                    t.join()
            vote_answers = []
            for j in range(len(answer_count)):
                vote_answer = None
                if len(answer_count[j]) > 0:
                    vote_answer = max(answer_count[j].items(), key=lambda x: x[1])[0]
                vote_answers.append(vote_answer)
            return x, vote_answers
        else:
            return x, None