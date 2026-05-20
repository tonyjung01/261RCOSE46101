import math

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm


LEGACY_VOTE_METHODS = {"fixed", "linear", "exp"}
CONFIDENCE_GAP_PREFIX = "confidence_gap_"


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


def parse_vote_method_config(vote_method, alpha=None, skip_first_ratio=0.0):
    if not 0.0 <= skip_first_ratio < 1.0:
        raise ValueError("skip_first_ratio must satisfy 0.0 <= skip_first_ratio < 1.0.")

    if vote_method in LEGACY_VOTE_METHODS:
        if vote_method == "exp" and alpha is None:
            raise ValueError("When vote_method='exp', alpha parameter must be provided.")
        return {
            "name": vote_method,
            "family": "temporal",
            "scale": "rawsum",
            "alpha": alpha,
            "skip_first_ratio": skip_first_ratio,
        }

    if not vote_method.startswith(CONFIDENCE_GAP_PREFIX):
        raise ValueError(
            "Unsupported vote_method. Expected one of "
            f"{sorted(LEGACY_VOTE_METHODS)} or a '{CONFIDENCE_GAP_PREFIX}...' method."
        )

    remainder = vote_method[len(CONFIDENCE_GAP_PREFIX) :]
    parts = remainder.split("_")
    if len(parts) not in {4, 5, 6}:
        raise ValueError(
            "Confidence-gap vote methods must follow "
            "'confidence_gap_<region>_<window>_<reduce>_<scale>' or "
            "'confidence_gap_<region>_<window>_<activity>_<reduce>_<scale>' or "
            "'confidence_gap_<region>_<window>_<activity>_<source>_<reduce>_<scale>'."
        )

    if len(parts) == 4:
        region, window_token, reduce_method, scale_method = parts
        activity_mode = "all"
        source_mode = "logit"
    elif len(parts) == 5:
        region, window_token, third_part, reduce_method, scale_method = parts
        if third_part in {"all", "active", "blockactive"}:
            activity_mode = third_part
            source_mode = "logit"
        elif third_part in {"logit", "prob"}:
            activity_mode = "all"
            source_mode = third_part
        else:
            raise ValueError(
                "Five-part confidence-gap methods must use either an activity "
                "token {'all','active','blockactive'} or a source token {'logit','prob'}."
            )
    else:
        region, window_token, activity_mode, source_mode, reduce_method, scale_method = parts
    if region not in {"answer", "anchor"}:
        raise ValueError("Confidence-gap region must be one of {'answer', 'anchor'}.")
    if not window_token.startswith("window"):
        raise ValueError("Confidence-gap window must look like 'window5'.")

    try:
        window_size = int(window_token[len("window") :])
    except ValueError as exc:
        raise ValueError("Confidence-gap window size must be an integer, e.g. window5.") from exc

    if window_size <= 0:
        raise ValueError("Confidence-gap window size must be positive.")
    if activity_mode not in {"all", "active", "blockactive"}:
        raise ValueError("Confidence-gap activity must be one of {'all', 'active', 'blockactive'}.")
    if source_mode not in {"logit", "prob"}:
        raise ValueError("Confidence-gap source must be one of {'logit', 'prob'}.")
    if reduce_method not in {"mean", "max", "min"}:
        raise ValueError("Confidence-gap reduce must be one of {'mean', 'max', 'min' }.")
    if scale_method not in {"rawsum", "stepsum1", "tanh"}:
        raise ValueError("Confidence-gap scale must be one of {'rawsum', 'stepsum1', 'tanh'}.")

    return {
        "name": vote_method,
        "family": "confidence_gap",
        "source": "top1_prob_minus_top2_prob" if source_mode == "prob" else "top1_logit_minus_top2_logit",
        "region": region,
        "window": window_token,
        "window_size": window_size,
        "activity": activity_mode,
        "reduce": reduce_method,
        "scale": scale_method,
        "alpha": alpha,
        "skip_first_ratio": skip_first_ratio,
    }


def _to_jsonable_answer(answer):
    if isinstance(answer, (np.integer, np.floating)):
        return answer.item()
    return answer


def _build_answer_candidates(parsed_answer):
    parsed_answer = _to_jsonable_answer(parsed_answer)
    if parsed_answer is None:
        return []

    candidates = []
    if isinstance(parsed_answer, int):
        candidates.append(str(parsed_answer))
    elif isinstance(parsed_answer, float):
        candidates.append(format(parsed_answer, "g"))
        candidates.append(str(parsed_answer))
        if parsed_answer.is_integer():
            candidates.append(str(int(parsed_answer)))
    else:
        answer_text = str(parsed_answer).strip()
        if answer_text:
            candidates.append(answer_text)
            compact_text = "".join(answer_text.split())
            if compact_text != answer_text:
                candidates.append(compact_text)

    expanded = []
    for candidate in candidates:
        if candidate:
            expanded.append(candidate)
            expanded.append(" " + candidate)

    deduped = []
    seen = set()
    for candidate in expanded:
        if candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped


def _find_last_subsequence(sequence, pattern):
    if not pattern or len(pattern) > len(sequence):
        return None

    match_start = None
    last_possible = len(sequence) - len(pattern)
    for start_idx in range(last_possible + 1):
        if sequence[start_idx : start_idx + len(pattern)] == pattern:
            match_start = start_idx
    return match_start


def _locate_answer_window(tokenizer, suffix_token_ids, parsed_answer, window_size):
    # Known limitation: candidate answers are tokenized in isolation before subsequence search.
    # BPE merges with surrounding context can therefore cause safe false negatives
    # (answer_window_not_found). We intentionally keep the current skip-on-miss behavior for now
    # and only revisit it if the skip rate remains high, especially on math500.
    best_match = None
    for candidate in _build_answer_candidates(parsed_answer):
        candidate_token_ids = tokenizer.encode(candidate, add_special_tokens=False)
        start_idx = _find_last_subsequence(suffix_token_ids, candidate_token_ids)
        if start_idx is None:
            continue

        match_info = {
            "matched_candidate": candidate,
            "answer_match_start": start_idx,
            "answer_match_token_length": len(candidate_token_ids),
            "window_start": start_idx,
            "window_end": min(start_idx + window_size, len(suffix_token_ids)),
            "region_kind": "answer_window",
        }
        if best_match is None:
            best_match = match_info
            continue

        if (
            match_info["answer_match_start"] > best_match["answer_match_start"]
            or (
                match_info["answer_match_start"] == best_match["answer_match_start"]
                and match_info["answer_match_token_length"] > best_match["answer_match_token_length"]
            )
        ):
            best_match = match_info

    return best_match


def _locate_anchor_window(prompt_length, suffix_length, answer_start_offset, window_size):
    if answer_start_offset is None:
        return None

    suffix_length = int(suffix_length)
    answer_start_offset = int(answer_start_offset)
    if answer_start_offset < 0 or answer_start_offset >= suffix_length:
        return None

    window_end = min(answer_start_offset + window_size, suffix_length)
    return {
        "region_kind": "anchor_window",
        "window_start": answer_start_offset,
        "window_end": window_end,
        "anchor_answer_start_offset": answer_start_offset,
        "anchor_answer_start_pos": prompt_length + answer_start_offset,
    }


def _reduce_gap_values(gap_tensor, reduce_method):
    if reduce_method == "mean":
        return float(gap_tensor.mean().item())
    if reduce_method == "max":
        return float(gap_tensor.max().item())
    if reduce_method == "min":
        return float(gap_tensor.min().item())
    raise ValueError(f"Unsupported reduce_method: {reduce_method}")


def _select_region_logits(region_logits, region_mask, absolute_positions, active_start_idx, active_end_idx, vote_config):
    answer_window_token_count = int(region_logits.shape[0])
    active_token_count = int(region_mask.sum().item())
    frozen_token_count = answer_window_token_count - active_token_count
    active_block_mask = (absolute_positions >= active_start_idx) & (absolute_positions < active_end_idx)
    active_block_overlap_token_count = int(active_block_mask.sum().item())

    selection_info = {
        "answer_window_token_count": answer_window_token_count,
        "active_token_count": active_token_count,
        "frozen_token_count": frozen_token_count,
        "active_block_overlap_token_count": active_block_overlap_token_count,
        "activity_mode": vote_config["activity"],
    }

    if vote_config["activity"] == "blockactive":
        if active_block_overlap_token_count == 0:
            selection_info["selected_token_count"] = 0
            return None, selection_info, "answer_window_outside_active_block"

        selected_mask = active_block_mask & region_mask
        selected_token_count = int(selected_mask.sum().item())
        selection_info["selected_token_count"] = selected_token_count
        if selected_token_count == 0:
            return None, selection_info, "answer_window_active_block_frozen"
        return region_logits[selected_mask], selection_info, None

    if vote_config["activity"] == "active":
        selection_info["selected_token_count"] = active_token_count
        if active_token_count == 0:
            return None, selection_info, "answer_window_frozen"
        return region_logits[region_mask], selection_info, None

    selection_info["selected_token_count"] = answer_window_token_count
    return region_logits, selection_info, None


def _compute_confidence_gap(selected_region_logits, vote_config):
    source = vote_config["source"]
    if source == "top1_prob_minus_top2_prob":
        probs = F.softmax(selected_region_logits.to(torch.float32), dim=-1)
        top2_vals, _ = torch.topk(probs, k=2, dim=-1)
        return (top2_vals[:, 0] - top2_vals[:, 1]).to(torch.float32)

    if source == "top1_logit_minus_top2_logit":
        top2_vals, _ = torch.topk(selected_region_logits, k=2, dim=-1)
        return (top2_vals[:, 0] - top2_vals[:, 1]).to(torch.float32)

    raise ValueError(f"Unsupported confidence-gap source: {source}")


def _legacy_vote_weight(step, steps, vote_config):
    if vote_config["name"] == "fixed":
        return 1.0
    if vote_config["name"] == "linear":
        return (step + 1) / steps
    if vote_config["name"] == "exp":
        return float(np.exp(step / steps * vote_config["alpha"]))
    raise ValueError(f"Unsupported legacy vote method: {vote_config['name']}")


def _scale_vote_weight(raw_weight, vote_config, normalization_denom):
    scale = vote_config["scale"]
    if scale == "rawsum":
        return raw_weight
    if scale == "stepsum1":
        if normalization_denom <= 0:
            return 0.0
        return raw_weight / normalization_denom
    if scale == "tanh":
        return math.tanh(raw_weight)
    raise ValueError(f"Unsupported vote scale: {scale}")


def _finalize_vote_events(vote_events, vote_config, save_vote_debug):
    vote_answers = []
    vote_debug = []

    for sample_events in vote_events:
        skip_counts = {}
        valid_events = []
        for event in sample_events:
            skip_reason = event.get("skip_reason")
            if skip_reason is None and event.get("raw_weight") is not None and event.get("parsed_answer") is not None:
                valid_events.append(event)
            else:
                skip_counts[skip_reason or "unknown"] = skip_counts.get(skip_reason or "unknown", 0) + 1

        normalization_denom = sum(event["raw_weight"] for event in valid_events)
        answer_scores = {}
        for event in valid_events:
            scaled_weight = _scale_vote_weight(event["raw_weight"], vote_config, normalization_denom)
            event["scaled_weight"] = scaled_weight
            answer_key = event["parsed_answer"]
            answer_scores[answer_key] = answer_scores.get(answer_key, 0.0) + scaled_weight

        sorted_scores = sorted(answer_scores.items(), key=lambda item: item[1], reverse=True)
        vote_answer = sorted_scores[0][0] if sorted_scores else None
        vote_answers.append(vote_answer)

        sample_summary = {
            "vote_method": vote_config["name"],
            "vote_config": dict(vote_config),
            "total_events": len(sample_events),
            "valid_events": len(valid_events),
            "skipped_events": len(sample_events) - len(valid_events),
            "skip_counts": skip_counts,
            "final_scores": [{"answer": answer, "score": score} for answer, score in sorted_scores],
            "vote_answer": vote_answer,
        }
        if save_vote_debug:
            sample_summary["steps"] = sample_events
        vote_debug.append(sample_summary)

    return vote_answers, vote_debug


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
    # === decoding policy ablation ===
    transfer_score="top1_prob",
    temporal_lambda=0.0,
    temporal_tau=0.15,
    # === vote related parameter ===
    enable_vote=False,
    tokenizer=None,
    parse_answer_func=None,
    vote_method=None,
    alpha=None,
    save_vote_debug=False,
    vote_skip_first_ratio=0.0,
    constraints=None,
    answer_start_offset=None,
    answer_length=None,
):
    """
    Optimized version of the generate function.
    """
    if enable_vote:
        if parse_answer_func is None or not callable(parse_answer_func):
            raise ValueError("When enable_vote=True, parse_answer_func must be provided and callable.")
        if tokenizer is None:
            raise ValueError("When enable_vote=True, tokenizer must be provided.")
        vote_config = parse_vote_method_config(vote_method, alpha=alpha, skip_first_ratio=vote_skip_first_ratio)
        if vote_config["family"] == "confidence_gap" and vote_config["region"] == "anchor":
            if answer_start_offset is None:
                raise ValueError(
                    "Anchor-based confidence-gap voting requires answer_start_offset to be provided."
                )
            if answer_length is not None and int(answer_length) != vote_config["window_size"]:
                raise ValueError(
                    "Anchor-based confidence-gap voting expects answer_length to match the method window size."
                )
        vote_config["start_step"] = math.ceil(steps * vote_skip_first_ratio)
        vote_events = [[] for _ in range(prompt.shape[0])]
    else:
        tokenizer = None
        parse_answer_func = None
        vote_method = None
        alpha = None
        vote_config = None
        vote_events = None

    # Use mixed precision for faster computation
    with torch.autocast(device_type="cuda"):
        x = torch.full(
            (prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=prompt.device
        )
        x[:, : prompt.shape[1]] = prompt.clone()

        if constraints is not None:
            for pos, token_id in constraints.items():
                absolute_pos = prompt.shape[1] + pos
                if absolute_pos < x.shape[1]:
                    x[:, absolute_pos] = token_id

        prompt_index = x != mask_id

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        steps_per_block = max(1, steps // num_blocks)
        for num_block in tqdm(range(num_blocks), disable=(dist.get_rank() != 0)):
            start_idx = prompt.shape[1] + num_block * block_length
            end_idx = prompt.shape[1] + (num_block + 1) * block_length

            block_mask_index = x[:, start_idx:end_idx] == mask_id
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

            use_temporal = transfer_score in {"temporal_margin", "gated_temporal_margin"}
            if use_temporal:
                prev_top1 = torch.full(x.shape, -1, dtype=torch.long, device=x.device)
                runlen = torch.zeros(x.shape, dtype=torch.float32, device=x.device)

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

                # Temporal state update: track consecutive top-1 identity per position.
                # Runs only for temporal_margin; zero overhead for top1_prob / prob_margin.
                if use_temporal:
                    same = (x0 == prev_top1)
                    runlen = torch.where(same, runlen + 1.0, torch.ones_like(runlen))
                    prev_top1 = x0.clone()

                # Handle remasking strategy
                if remasking == "low_confidence":
                    # Use float32 instead of float64 for better performance
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                    # Decoding policy ablation: override the ranking score
                    # for token-transfer selection. Filled token (`x0`) is
                    # unchanged; only the position-selection ranking changes.
                    if transfer_score == "top1_prob":
                        pass  # baseline: x0_p == p_top1 (since x0 = argmax(logits) at T=0)
                    elif transfer_score == "prob_margin":
                        top2_vals, _ = p.topk(k=2, dim=-1)
                        x0_p = top2_vals[..., 0] - top2_vals[..., 1]
                    elif transfer_score == "temporal_margin":
                        # C1: Kim-style margin + block-normalized run-length stability.
                        # stability ∈ (0, 1]: fraction of the current block's steps
                        # for which this position's top-1 token has been unchanged.
                        top2_vals, _ = p.topk(k=2, dim=-1)
                        margin = top2_vals[..., 0] - top2_vals[..., 1]
                        stability = runlen / float(i + 1)
                        x0_p = margin + temporal_lambda * stability
                    elif transfer_score == "gated_temporal_margin":
                        # C-next-1: keep strong margin decisions intact and only
                        # apply temporal stability to ambiguous positions.
                        top2_vals, _ = p.topk(k=2, dim=-1)
                        margin = top2_vals[..., 0] - top2_vals[..., 1]
                        stability = runlen / float(i + 1)
                        ambiguous = (margin < temporal_tau).to(stability.dtype)
                        x0_p = margin + temporal_lambda * stability * ambiguous
                    else:
                        raise ValueError(f"Unsupported transfer_score: {transfer_score}")
                elif remasking == "random":
                    x0_p = torch.rand(x0.shape, device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                # Ensure we don't process tokens beyond the current block
                x0_p[:, end_idx:] = -np.inf

                # Update masked tokens
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x0.device))

                # Parse answer if voting is enabled.
                if enable_vote:
                    generated_texts = tokenizer.batch_decode(x0[:, prompt.shape[1] :], skip_special_tokens=True)
                    for j, generated_text in enumerate(generated_texts):
                        event = {
                            "step": step,
                            "total_steps": steps,
                            "parsed_answer": None,
                            "vote_method": vote_config["name"],
                            "skip_reason": None,
                        }

                        if step < vote_config["start_step"]:
                            event["skip_reason"] = "warmup_step_skipped"
                            vote_events[j].append(event)
                            continue

                        parsed_answer = _to_jsonable_answer(parse_answer_func(generated_text))
                        event["parsed_answer"] = parsed_answer

                        if parsed_answer is None:
                            event["skip_reason"] = "parse_failed"
                            vote_events[j].append(event)
                            continue

                        if vote_config["family"] == "temporal":
                            event["raw_weight"] = _legacy_vote_weight(step, steps, vote_config)
                            event["weight_source"] = "temporal"
                            vote_events[j].append(event)
                            continue

                        if vote_config["region"] == "anchor":
                            region = _locate_anchor_window(
                                prompt_length=prompt.shape[1],
                                suffix_length=x0.shape[1] - prompt.shape[1],
                                answer_start_offset=answer_start_offset,
                                window_size=vote_config["window_size"],
                            )
                            if region is None:
                                event["skip_reason"] = "anchor_window_out_of_range"
                                vote_events[j].append(event)
                                continue
                        else:
                            suffix_token_ids = x0[j, prompt.shape[1] :].tolist()
                            region = _locate_answer_window(
                                tokenizer=tokenizer,
                                suffix_token_ids=suffix_token_ids,
                                parsed_answer=parsed_answer,
                                window_size=vote_config["window_size"],
                            )
                        if region is None:
                            event["skip_reason"] = "answer_window_not_found"
                            vote_events[j].append(event)
                            continue

                        abs_start = prompt.shape[1] + region["window_start"]
                        abs_end = prompt.shape[1] + region["window_end"]
                        region_logits = logits[j, abs_start:abs_end, :]
                        region_mask = mask_index[j, abs_start:abs_end]
                        absolute_positions = torch.arange(abs_start, abs_end, device=region_logits.device)
                        selected_region_logits, selection_info, selection_skip_reason = _select_region_logits(
                            region_logits=region_logits,
                            region_mask=region_mask,
                            absolute_positions=absolute_positions,
                            active_start_idx=start_idx,
                            active_end_idx=end_idx,
                            vote_config=vote_config,
                        )
                        event.update(region)
                        event.update(selection_info)
                        if selection_skip_reason is not None:
                            event["skip_reason"] = selection_skip_reason
                            vote_events[j].append(event)
                            continue

                        event["gap_source"] = vote_config["source"]
                        event["gap_reduce"] = vote_config["reduce"]
                        gap_tensor = _compute_confidence_gap(selected_region_logits, vote_config)
                        event["raw_weight"] = _reduce_gap_values(gap_tensor, vote_config["reduce"])
                        if save_vote_debug:
                            event["gap_values"] = [float(value) for value in gap_tensor.detach().cpu().tolist()]
                        vote_events[j].append(event)

                # Select tokens to transfer based on confidence
                for j in range(confidence.shape[0]):
                    num_tokens = num_transfer_tokens[j, i].item()
                    if num_tokens > 0:
                        _, select_indices = torch.topk(confidence[j], k=num_tokens)
                        x[j, select_indices] = x0[j, select_indices]
                        if enable_vote and save_vote_debug and vote_events[j]:
                            vote_events[j][-1]["selected_transfer_indices"] = select_indices.cpu().tolist()

                if constraints is not None:
                    for pos, token_id in constraints.items():
                        absolute_pos = prompt.shape[1] + pos
                        if absolute_pos < x.shape[1]:
                            x[:, absolute_pos] = token_id

        if enable_vote:
            vote_answers, vote_debug = _finalize_vote_events(vote_events, vote_config, save_vote_debug)
            return x, vote_answers, vote_debug
        return x, None, None
