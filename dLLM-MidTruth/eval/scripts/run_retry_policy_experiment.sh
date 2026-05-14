#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EVAL_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$EVAL_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

DEFAULT_HF_HOME="/home/work/GFlowPO/jaeyoon/.cache/huggingface"
DEFAULT_HF_DATASETS_CACHE="/home/work/GFlowPO/jaeyoon/hf-cache/datasets"
DEFAULT_LOCAL_MODEL_PATH="/home/work/GFlowPO/jaeyoon/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07"

if [ -d "$DEFAULT_HF_HOME" ]; then
  export HF_HOME="${HF_HOME:-$DEFAULT_HF_HOME}"
fi

if [ -d "$DEFAULT_HF_DATASETS_CACHE" ]; then
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$DEFAULT_HF_DATASETS_CACHE}"
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

if [ "${HF_HUB_OFFLINE}" = "1" ]; then
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
fi

read -r -a GPU_IDS <<< "${GPU_IDS:-0}"
if [ "$#" -gt 0 ]; then
  GPU_IDS=("$@")
fi

TASK="${TASK:-gsm8k}"
if [ -d "$DEFAULT_LOCAL_MODEL_PATH" ]; then
  MODEL_PATH="${MODEL_PATH:-$DEFAULT_LOCAL_MODEL_PATH}"
else
  MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
fi
MODEL_NAME="${MODEL_NAME:-LLaDA-8B-Instruct}"
MASTER_PORT="${MASTER_PORT:-29415}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GEN_LENGTH="${GEN_LENGTH:-128}"
TOKEN_PER_STEP="${TOKEN_PER_STEP:-2}"
BLOCK_LENGTH="${BLOCK_LENGTH:-32}"
TEMPERATURE="${TEMPERATURE:-0.0}"
VOTE_METHOD="${VOTE_METHOD:-exp}"
ALPHA="${ALPHA:-5.0}"
SAVE_VOTE_DEBUG="${SAVE_VOTE_DEBUG:-true}"
SUBSET_INDICES_FILE="${SUBSET_INDICES_FILE:-}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)_retry_${TASK}_${VOTE_METHOD}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${MODEL_NAME}/${RUN_NAME}}"

if [ -z "$SUBSET_INDICES_FILE" ]; then
  echo "SUBSET_INDICES_FILE is required"
  exit 1
fi

if [ ! -f "$SUBSET_INDICES_FILE" ]; then
  echo "SUBSET_INDICES_FILE does not exist: $SUBSET_INDICES_FILE"
  exit 1
fi

if [[ "$MODEL_PATH" = /* ]]; then
  if [ ! -d "$MODEL_PATH" ]; then
    echo "MODEL_PATH directory does not exist: $MODEL_PATH"
    exit 1
  fi
else
  if [ "${HF_HUB_OFFLINE}" = "1" ]; then
    echo "MODEL_PATH is a Hub repo id but HF_HUB_OFFLINE=1: $MODEL_PATH"
    echo "Set MODEL_PATH to a local snapshot path for offline selective retry runs."
    exit 1
  fi
fi

if [ "$TASK" = "gsm8k" ] && [ "${HF_HUB_OFFLINE}" = "1" ] && [ -n "${HF_DATASETS_CACHE:-}" ] && [ ! -d "${HF_DATASETS_CACHE}/openai___gsm8k" ]; then
  echo "HF_DATASETS_CACHE does not contain gsm8k cache: ${HF_DATASETS_CACHE}/openai___gsm8k"
  exit 1
fi

if [ -e "$OUTPUT_DIR/rank_0_generations.json" ]; then
  echo "Output file already exists: $OUTPUT_DIR/rank_0_generations.json"
  echo "Choose a different RUN_NAME or OUTPUT_DIR to avoid overwriting a retry run."
  exit 1
fi

GPU_LIST=$(IFS=,; echo "${GPU_IDS[*]}")
NUM_GPUS=${#GPU_IDS[@]}
DIFFUSION_STEPS=$((GEN_LENGTH / TOKEN_PER_STEP))

mkdir -p "$OUTPUT_DIR"
TORCHRUN_LOG="$OUTPUT_DIR/torchrun_console.log"

echo "Running preflight checks"
echo "  task=$TASK"
echo "  python_bin=$PYTHON_BIN"
echo "  hf_home=${HF_HOME:-<unset>}"
echo "  hf_datasets_cache=${HF_DATASETS_CACHE:-<unset>}"
echo "  hf_hub_offline=$HF_HUB_OFFLINE"
echo "  model_path=$MODEL_PATH"

MODEL_PATH="$MODEL_PATH" "$PYTHON_BIN" - <<'PY'
import os
from transformers import AutoConfig

model_path = os.environ["MODEL_PATH"]
AutoConfig.from_pretrained(model_path, trust_remote_code=True)
print(f"[preflight] model config load OK: {model_path}")
PY

if [ "$TASK" = "gsm8k" ]; then
  "$PYTHON_BIN" - <<'PY'
from datasets import load_dataset

dataset = load_dataset("openai/gsm8k", "main", split="test")
print(f"[preflight] gsm8k load OK: {len(dataset)} rows")
PY
fi

CMD=(
  "$PYTHON_BIN"
  -m
  torch.distributed.run
  --nproc_per_node "$NUM_GPUS"
  --master_port "$MASTER_PORT"
  eval.py
  --dataset "$TASK"
  --batch_size "$BATCH_SIZE"
  --gen_length "$GEN_LENGTH"
  --diffusion_steps "$DIFFUSION_STEPS"
  --block_length "$BLOCK_LENGTH"
  --temperature "$TEMPERATURE"
  --output_dir "$OUTPUT_DIR"
  --model_path "$MODEL_PATH"
  --model_name "$MODEL_NAME"
  --subset_indices_file "$SUBSET_INDICES_FILE"
  --enable_vote
  --vote_method "$VOTE_METHOD"
)

if [ "$VOTE_METHOD" = "exp" ]; then
  CMD+=(--alpha "$ALPHA")
fi

if [ "$SAVE_VOTE_DEBUG" = "true" ] || [ "$SAVE_VOTE_DEBUG" = "1" ]; then
  CMD+=(--save_vote_debug)
fi

echo "Running selective retry rerun"
echo "  task=$TASK"
echo "  vote_method=$VOTE_METHOD"
echo "  python_bin=$PYTHON_BIN"
echo "  hf_home=${HF_HOME:-<unset>}"
echo "  hf_datasets_cache=${HF_DATASETS_CACHE:-<unset>}"
echo "  hf_hub_offline=$HF_HUB_OFFLINE"
echo "  model_path=$MODEL_PATH"
echo "  subset_indices_file=$SUBSET_INDICES_FILE"
echo "  output_dir=$OUTPUT_DIR"
echo "  gpus=$GPU_LIST"
echo "  torchrun_log=$TORCHRUN_LOG"

CUDA_VISIBLE_DEVICES="$GPU_LIST" "${CMD[@]}" 2>&1 | tee "$TORCHRUN_LOG"
"$PYTHON_BIN" get_acc.py "$OUTPUT_DIR" | tee "$OUTPUT_DIR/accuracy.txt"

echo "Done."
