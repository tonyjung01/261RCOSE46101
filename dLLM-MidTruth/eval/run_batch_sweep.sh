#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

CONDA_SH="${CONDA_SH:-/home/work/GFlowPO/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-tiaf}"
if [ -f "$CONDA_SH" ]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
fi

DATASET_KEY="${DATASET_KEY:-gsm8k}"
DATASET_LABEL="${DATASET_LABEL:-$DATASET_KEY}"
GPU_ID="${GPU_ID:-0}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29413}"
MODEL_NAME="${MODEL_NAME:-LLaDA-8B-Instruct}"
MODEL_PATH="${MODEL_PATH:-/home/work/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07}"
ENABLE_VOTE="${ENABLE_VOTE:-true}"
VOTE_METHOD="${VOTE_METHOD:-exp}"
ALPHA="${ALPHA:-5.0}"
GEN_LENGTHS="${GEN_LENGTHS:-128}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4 6 8 10 12 14 16}"
SWEEP_NAME="${SWEEP_NAME:-${DATASET_LABEL}_bs_sweep_$(date +%Y%m%d_%H%M%S)}"
SWEEP_ROOT="outputs/${MODEL_NAME}/${SWEEP_NAME}"

mkdir -p "$SWEEP_ROOT"

cat > "${SWEEP_ROOT}/sweep_config.txt" <<CFG
DATASET_KEY=$DATASET_KEY
DATASET_LABEL=$DATASET_LABEL
GPU_ID=$GPU_ID
MASTER_PORT_BASE=$MASTER_PORT_BASE
MODEL_NAME=$MODEL_NAME
MODEL_PATH=$MODEL_PATH
ENABLE_VOTE=$ENABLE_VOTE
VOTE_METHOD=$VOTE_METHOD
ALPHA=$ALPHA
GEN_LENGTHS=$GEN_LENGTHS
BATCH_SIZES=$BATCH_SIZES
SWEEP_NAME=$SWEEP_NAME
CFG

echo "Starting batch sweep"
echo "  dataset: $DATASET_KEY ($DATASET_LABEL)"
echo "  gpu: $GPU_ID"
echo "  model: $MODEL_NAME"
echo "  gen_lengths: $GEN_LENGTHS"
echo "  batch_sizes: $BATCH_SIZES"
echo "  sweep_root: $SWEEP_ROOT"

port_offset=0
for batch_size in $BATCH_SIZES; do
  run_name="${SWEEP_NAME}_bs${batch_size}"
  output_root="${SWEEP_ROOT}/bs${batch_size}"
  master_port=$((MASTER_PORT_BASE + port_offset))

  echo
  echo "=== Running ${DATASET_LABEL} batch_size=${batch_size} on GPU ${GPU_ID} (port ${master_port}) ==="

  RUN_NAME="$run_name" \
  OUTPUT_ROOT="$output_root" \
  TASKS="$DATASET_KEY" \
  GEN_LENGTHS="$GEN_LENGTHS" \
  GPU_IDS="$GPU_ID" \
  MASTER_PORT="$master_port" \
  MODEL_PATH="$MODEL_PATH" \
  MODEL_NAME="$MODEL_NAME" \
  ENABLE_VOTE="$ENABLE_VOTE" \
  VOTE_METHOD="$VOTE_METHOD" \
  ALPHA="$ALPHA" \
  BATCH_SIZE_OVERRIDE="$batch_size" \
  bash run_eval.sh

  port_offset=$((port_offset + 1))
done

echo
echo "Batch sweep finished: $SWEEP_ROOT"
