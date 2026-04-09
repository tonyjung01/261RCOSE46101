#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

read -r -a TASKS <<< "${TASKS:-countdown}"
read -r -a GEN_LENGTHS <<< "${GEN_LENGTHS:-128}"
read -r -a GPU_IDS <<< "${GPU_IDS:-0}"

if [ "$#" -gt 0 ]; then
  GPU_IDS=("$@")
fi

MASTER_PORT="${MASTER_PORT:-29413}"
TOKEN_PER_STEP="${TOKEN_PER_STEP:-2}"
BLOCK_LENGTH="${BLOCK_LENGTH:-32}"
TEMPERATURE="${TEMPERATURE:-0.0}"

MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
MODEL_NAME="${MODEL_NAME:-LLaDA-8B-Instruct}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/${MODEL_NAME}/${RUN_NAME}}"

ENABLE_VOTE="${ENABLE_VOTE:-true}"
VOTE_METHOD="${VOTE_METHOD:-exp}"
ALPHA="${ALPHA:-5.0}"

RUN_GET_ACC="${RUN_GET_ACC:-true}"
BATCH_SIZE_OVERRIDE="${BATCH_SIZE_OVERRIDE:-}"

is_true() {
  case "$1" in
    true|TRUE|1|yes|YES|y|Y) return 0 ;;
    *) return 1 ;;
  esac
}

default_batch_size() {
  local task="$1"
  local gen_length="$2"
  local batch_size

  if [ -n "$BATCH_SIZE_OVERRIDE" ]; then
    echo "$BATCH_SIZE_OVERRIDE"
    return
  fi

  if [ "$gen_length" = "256" ] || [ "$gen_length" = "512" ]; then
    batch_size=4
  else
    batch_size=8
  fi

  if [ "$task" = "math" ]; then
    batch_size=$((batch_size / 4))
  fi

  if [ "$batch_size" -lt 1 ]; then
    batch_size=1
  fi

  echo "$batch_size"
}

build_mode_tag() {
  if is_true "$ENABLE_VOTE"; then
    echo "vote_${VOTE_METHOD}_a${ALPHA}"
  else
    echo "novote"
  fi
}

if [ -e "$OUTPUT_ROOT" ]; then
  echo "Output root already exists: $OUTPUT_ROOT"
  echo "Set a different RUN_NAME or OUTPUT_ROOT to keep experiments separated."
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

GPU_LIST=$(IFS=,; echo "${GPU_IDS[*]}")
NUM_GPUS=${#GPU_IDS[@]}
MODE_TAG=$(build_mode_tag)

RUN_CONFIG_FILE="$OUTPUT_ROOT/run_config.txt"
printf '%s\n' \
  "MODEL_PATH=$MODEL_PATH" \
  "MODEL_NAME=$MODEL_NAME" \
  "RUN_NAME=$RUN_NAME" \
  "OUTPUT_ROOT=$OUTPUT_ROOT" \
  "TASKS=${TASKS[*]}" \
  "GEN_LENGTHS=${GEN_LENGTHS[*]}" \
  "GPU_IDS=${GPU_IDS[*]}" \
  "MASTER_PORT=$MASTER_PORT" \
  "BLOCK_LENGTH=$BLOCK_LENGTH" \
  "TOKEN_PER_STEP=$TOKEN_PER_STEP" \
  "TEMPERATURE=$TEMPERATURE" \
  "ENABLE_VOTE=$ENABLE_VOTE" \
  "VOTE_METHOD=$VOTE_METHOD" \
  "ALPHA=$ALPHA" \
  "RUN_GET_ACC=$RUN_GET_ACC" \
  "BATCH_SIZE_OVERRIDE=$BATCH_SIZE_OVERRIDE" \
  > "$RUN_CONFIG_FILE"

echo "Using GPUs: $GPU_LIST (nproc_per_node=$NUM_GPUS)"
echo "Run name: $RUN_NAME"
echo "Output root: $OUTPUT_ROOT"

for task in "${TASKS[@]}"; do
  for gen_length in "${GEN_LENGTHS[@]}"; do
    batch_size=$(default_batch_size "$task" "$gen_length")
    diffusion_steps=$((gen_length / TOKEN_PER_STEP))
    output_dir="$OUTPUT_ROOT/${task}_gen_${gen_length}_steps_${diffusion_steps}_temp_${TEMPERATURE}_${MODE_TAG}_bs${batch_size}"

    if [ -e "$output_dir" ]; then
      echo "Output directory already exists: $output_dir"
      echo "Choose a new RUN_NAME to avoid mixing runs."
      exit 1
    fi

    echo "Running task=$task gen_length=$gen_length diffusion_steps=$diffusion_steps batch_size=$batch_size output_dir=$output_dir"

    cmd=(
      torchrun
      --nproc_per_node "$NUM_GPUS"
      --master_port "$MASTER_PORT"
      eval.py
      --dataset "$task"
      --batch_size "$batch_size"
      --gen_length "$gen_length"
      --diffusion_steps "$diffusion_steps"
      --block_length "$BLOCK_LENGTH"
      --temperature "$TEMPERATURE"
      --output_dir "$output_dir"
      --model_path "$MODEL_PATH"
      --model_name "$MODEL_NAME"
    )

    if is_true "$ENABLE_VOTE"; then
      cmd+=(
        --enable_vote
        --vote_method "$VOTE_METHOD"
        --alpha "$ALPHA"
      )
    fi

    CUDA_VISIBLE_DEVICES="$GPU_LIST" "${cmd[@]}"

    if is_true "$RUN_GET_ACC"; then
      python get_acc.py "$output_dir" | tee "$output_dir/accuracy.txt"
    fi

    echo
  done
done

if is_true "$RUN_GET_ACC"; then
  python get_acc.py "$OUTPUT_ROOT" | tee "$OUTPUT_ROOT/summary.txt"
fi

echo "All evaluations completed."
