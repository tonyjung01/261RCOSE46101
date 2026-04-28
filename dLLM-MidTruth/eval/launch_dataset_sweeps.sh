#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

LAUNCH_MODE="${LAUNCH_MODE:-tmux}"
TMUX_BIN="${TMUX_BIN:-tmux}"
SESSION_PREFIX="${SESSION_PREFIX:-dmid_bs_$(date +%m%d_%H%M%S)}"
MODEL_NAME="${MODEL_NAME:-LLaDA-8B-Instruct}"
MODEL_PATH="${MODEL_PATH:-/home/work/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07}"
ENABLE_VOTE="${ENABLE_VOTE:-true}"
VOTE_METHOD="${VOTE_METHOD:-exp}"
ALPHA="${ALPHA:-5.0}"
GEN_LENGTHS="${GEN_LENGTHS:-128}"
GSM8K_BATCH_SIZES="${GSM8K_BATCH_SIZES:-1 2 4 6 8 10 12 14 16}"
MATH500_BATCH_SIZES="${MATH500_BATCH_SIZES:-1 2 4 6 8 10 12 14 16}"
SVAMP_BATCH_SIZES="${SVAMP_BATCH_SIZES:-1 2 4 6 8 10 12 14 16}"

declare -a DATASET_KEYS=("gsm8k" "math" "svamp")
declare -a DATASET_LABELS=("gsm8k" "math500" "svamp")
declare -a DATASET_BATCH_SIZES=("$GSM8K_BATCH_SIZES" "$MATH500_BATCH_SIZES" "$SVAMP_BATCH_SIZES")
declare -a GPU_IDS=("0" "2" "3")
declare -a MASTER_PORT_BASES=("29413" "29513" "29613")

if [ "$LAUNCH_MODE" = "tmux" ] && ! command -v "$TMUX_BIN" >/dev/null 2>&1; then
  echo "tmux is not installed on this machine."
  echo "Install tmux or rerun with LAUNCH_MODE=nohup."
  exit 1
fi

mkdir -p logs

for idx in "${!DATASET_KEYS[@]}"; do
  dataset_key="${DATASET_KEYS[$idx]}"
  dataset_label="${DATASET_LABELS[$idx]}"
  batch_sizes="${DATASET_BATCH_SIZES[$idx]}"
  gpu_id="${GPU_IDS[$idx]}"
  master_port_base="${MASTER_PORT_BASES[$idx]}"
  sweep_name="${SESSION_PREFIX}_${dataset_label}"
  session_name="${SESSION_PREFIX}_${dataset_label}"
  log_path="logs/${session_name}.log"

  cmd="cd '$SCRIPT_DIR' && DATASET_KEY='$dataset_key' DATASET_LABEL='$dataset_label' GPU_ID='$gpu_id' MASTER_PORT_BASE='$master_port_base' MODEL_NAME='$MODEL_NAME' MODEL_PATH='$MODEL_PATH' ENABLE_VOTE='$ENABLE_VOTE' VOTE_METHOD='$VOTE_METHOD' ALPHA='$ALPHA' GEN_LENGTHS='$GEN_LENGTHS' BATCH_SIZES='$batch_sizes' SWEEP_NAME='$sweep_name' bash run_batch_sweep.sh"

  if [ "$LAUNCH_MODE" = "tmux" ]; then
    "$TMUX_BIN" new-session -d -s "$session_name" "$cmd"
    echo "Started tmux session: $session_name"
    echo "  attach: tmux attach -t $session_name"
    echo "  batch_sizes: $batch_sizes"
  else
    nohup bash -lc "$cmd" > "$log_path" 2>&1 &
    echo "Started nohup job for $dataset_label"
    echo "  log: $log_path"
    echo "  batch_sizes: $batch_sizes"
  fi
done
