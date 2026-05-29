# Temporal Reliability Token Ordering for Reasoning in Diffusion Language Models

**Korea University COSE461 Final Project — Team 6**  
Jaeyoon Jung · Byungmin Park · Eunseo Choi

---

## Base Work: Time Is a Feature (TIAF)

This project builds directly on **"Time Is a Feature: Exploiting Temporal Dynamics in Diffusion Language Models"** (Wang et al., 2025, arXiv:2508.09138).

TIAF identifies **temporal oscillation** in diffusion language models (dLLMs) — the phenomenon where correct answers can appear at intermediate denoising steps but get overwritten in later steps. To exploit this, the paper proposes:

- **Temporal Self-Consistency Voting (TSCV):** a training-free strategy that aggregates predictions across denoising steps using exponential weighting to select the most consistent answer.
- **Temporal Consistency Reinforcement:** a training method using Temporal Semantic Entropy (TSE) as a reward signal to encourage generation stability.

The `dLLM-MidTruth/` directory is a git clone of the TIAF codebase that we have customized for our experiments.

---

## Our Work

We study **temporal reliability as a token-ordering signal** in deterministic masked diffusion decoding.

Standard decoding policies commit tokens by ranking masked positions using instantaneous confidence scores (e.g., top-1 probability). We ask: can signals derived from how token predictions *evolve across denoising steps* improve these ranking decisions?

Building on **probability margin** (Kim et al., 2025) — the gap between top-1 and top-2 predicted probabilities — as our base ranking signal, we propose two training-free policies that modify only the token-position ranking rule at each denoising step:

- **GTTO (Gated Temporal Token Ordering):** Combines probability margin with top-1 temporal stability, but activates the stability term *only* when the current margin falls below an ambiguity threshold `τ`. Unambiguous positions remain governed by margin alone, while ambiguous positions are guided by recent top-1 token stability.
  > Score: `m(i,t) + λ · s(i,t) · 1[m(i,t) < τ]`

- **MargExKL (Margin with Exponential KL Penalty):** Penalizes margin using step-to-step KL divergence between consecutive predictive distributions, discounting positions whose distributions shift sharply over time.
  > Score: `m(i,t) · exp(−γ · KL(i,t))`

We evaluate on **LLaDA-8B-Instruct** and **LLaDA-1.5** across **GSM8K, MATH500, SVAMP, and Countdown**, using TSCV as the aggregation mechanism. Both methods improve over `top1_prob` across all four benchmarks and both models. GTTO is stronger on arithmetic word problems (GSM8K, SVAMP); MargExKL is stronger on tasks with more diverse reasoning trajectories (MATH500, Countdown).

---

## Contributions and Code Changes

All decoding policies are implemented as drop-in replacements within TIAF's semi-autoregressive decoding loop. No model weights, prompts, or answer parsers are modified.

### New token-ordering policies — `dLLM-MidTruth/eval/generate.py`

| Policy name (--transfer_score) | Description |
|---|---|
| `prob_margin` | Baseline: p_top1 − p_top2 (Kim et al.) |
| `temporal_margin` | Margin + λ·stability (always on; ablation only) |
| `gated_temporal_margin` | **GTTO**: margin + λ·stability·1[margin < τ] |
| `margin_exp_kl` | **MargExKL**: margin · exp(−γ·KL) |
| `exp_kl_decay` | KL-only ablation: exp(−γ·KL), no margin |

The `top1_prob` policy (original TIAF default) is unchanged.  
`prob_margin` margin computation, temporal run-length stability tracking (`use_temporal` block), and the truncated KL penalty (`use_kl` block) were coded by us.

### New evaluation arguments — `dLLM-MidTruth/eval/eval.py`

| Argument | Description |
|---|---|
| `--transfer_score` | Select a token-ordering policy (see table above) |
| `--temporal_lambda` | λ weight for temporal stability term (GTTO / temporal_margin) |
| `--temporal_tau` | τ ambiguity threshold for GTTO (default: 0.15) |
| `--kl_gamma` | γ KL penalty weight for MargExKL / exp_kl_decay |

---

## Prophet (Reference Only)

The `Prophet/` directory contains the code from **"Diffusion Language Models Know the Answer Before Decoding"** (Li et al., 2025, arXiv:2508.19982), included verbatim as a code reference. It was not modified and is not used in our experiments.

---

## Setup

### Requirements

- Python 3.10+
- 2× NVIDIA H100 (or equivalent; experiments were run on 2× H100)
- CUDA-compatible PyTorch environment

### Environment

```bash
conda env create -f dLLM-MidTruth/env.yml
conda activate tiaf
bash dLLM-MidTruth/install.sh
```

### Models

Download the base model checkpoints from HuggingFace and place them under `pretrained_weights/`:

- [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) → `pretrained_weights/GSAI-ML/LLaDA-8B-Instruct`
- [LLaDA-1.5](https://huggingface.co/GSAI-ML/LLaDA-1.5) → `pretrained_weights/GSAI-ML/LLaDA-1.5`

### Datasets

Datasets are loaded from the `dataset/` directory:

| Dataset | HuggingFace ID |
|---|---|
| GSM8K | `openai/gsm8k` |
| MATH500 | `ankner/math-500` |
| SVAMP | `ChilleD/SVAMP` |
| Countdown | `Jiayi-Pan/Countdown-Tasks-3to4` |

---

## Evaluation

All evaluation commands are run from inside `dLLM-MidTruth/eval/`.

### Baseline (`top1_prob`)

```bash
cd dLLM-MidTruth/eval

CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --nproc_per_node 2 \
    --master_port 29173 \
    eval.py \
    --dataset gsm8k \
    --batch_size 4 \
    --gen_length 128 \
    --diffusion_steps 64 \
    --output_dir "outputs/LLaDA-8B-Instruct/gsm8k_gen128_top1prob" \
    --model_path ../../pretrained_weights/GSAI-ML/LLaDA-8B-Instruct \
    --enable_vote \
    --vote_method exp \
    --alpha 5.0 \
    --transfer_score top1_prob
```

### GTTO (`gated_temporal_margin`)

Best configurations from our sweep: `(λ=0.10, τ=0.15)` for LLaDA-8B-Instruct; `(λ=0.05, τ=0.12)` for LLaDA-1.5.

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --nproc_per_node 2 \
    --master_port 29173 \
    eval.py \
    --dataset gsm8k \
    --batch_size 4 \
    --gen_length 128 \
    --diffusion_steps 64 \
    --output_dir "outputs/LLaDA-8B-Instruct/gsm8k_gen128_gtto" \
    --model_path ../../pretrained_weights/GSAI-ML/LLaDA-8B-Instruct \
    --enable_vote \
    --vote_method exp \
    --alpha 5.0 \
    --transfer_score gated_temporal_margin \
    --temporal_lambda 0.10 \
    --temporal_tau 0.15
```

### MargExKL (`margin_exp_kl`)

Best configurations from our sweep: `γ=0.5` for MATH500; `γ=0.3` for Countdown at longer lengths.

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --nproc_per_node 2 \
    --master_port 29173 \
    eval.py \
    --dataset math \
    --batch_size 4 \
    --gen_length 128 \
    --diffusion_steps 64 \
    --output_dir "outputs/LLaDA-8B-Instruct/math500_gen128_margexkl" \
    --model_path ../../pretrained_weights/GSAI-ML/LLaDA-8B-Instruct \
    --enable_vote \
    --vote_method exp \
    --alpha 5.0 \
    --transfer_score margin_exp_kl \
    --kl_gamma 0.5
```

### Computing Accuracy

After evaluation, run:

```bash
python get_acc.py --output_dir outputs/LLaDA-8B-Instruct/gsm8k_gen128_gtto
```

### Batch Sweeps

For convenience, `eval/run_eval.sh` and `eval/run_batch_sweep.sh` provide wrapper scripts for multi-configuration sweeps. Edit the parameters in these scripts to test different datasets, methods, and generation lengths.

---

## Key Hyperparameters

| Parameter | Value used |
|---|---|
| Block length | 32 |
| Tokens transferred per step | 2 |
| Temperature | 0.0 (greedy) |
| TSCV weight (α) | 5.0 |
| Global random seed | 42 |
| Per-process batch size | 4 |
| Distributed setup | 2× H100, `torchrun` |

---

## License

The `dLLM-MidTruth/` codebase is licensed under the [BSD 2-Clause License](https://opensource.org/license/bsd-2-clause) (inherited from TIAF). Our additions follow the same license.

---

## References

- Wang et al. (2025). *Time Is a Feature: Exploiting Temporal Dynamics in Diffusion Language Models.* arXiv:2508.09138.
- Kim et al. (2025). *Train-Free Masked Diffusion Language Model Decoding via Probability Margin.* (prob_margin baseline)
- Kim et al. (2025). *KLASS: KL-Adaptive Stability Sampling.* (MargExKL motivation)
- Nie et al. (2025). *LLaDA: Large Language Diffusion with mAsking.* (base model)
- Li et al. (2025). *Diffusion Language Models Know the Answer Before Decoding.* arXiv:2508.19982. (Prophet, reference only)
