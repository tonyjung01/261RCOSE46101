#!/usr/bin/env bash
set -euo pipefail
cd "/home/ubuntu/pbm/261RCOSE46101/dLLM-MidTruth/eval"

subset_for_task() {
  case "$1" in
    gsm8k) echo "/home/ubuntu/pbm/261RCOSE46101/dLLM-MidTruth/eval/analysis/transfer_score_probe/full_gsm8k_indices.txt" ;;
    svamp) echo "/home/ubuntu/pbm/261RCOSE46101/dLLM-MidTruth/eval/analysis/transfer_score_probe/full_svamp_indices.txt" ;;
    math|math500) echo "/home/ubuntu/pbm/261RCOSE46101/dLLM-MidTruth/eval/analysis/transfer_score_probe/full_math500_indices.txt" ;;
    countdown) echo "/home/ubuntu/pbm/261RCOSE46101/dLLM-MidTruth/eval/analysis/transfer_score_probe/full_countdown_indices.txt" ;;
    *) echo "Unknown task: $1" >&2; return 1 ;;
  esac
}

eval_task_name() {
  case "$1" in
    math500) echo "math" ;;
    *) echo "$1" ;;
  esac
}

run_prefix_for_task() {
  case "$1" in
    math) echo "20260521_math500_full_exp_bs4" ;;
    *) echo "20260521_${1}_full_exp_bs4" ;;
  esac
}

run_candidate() {
  local task_alias="$1"
  local gpu="$2"
  local port="$3"
  local task
  task=$(eval_task_name "$task_alias")
  local subset_file
  subset_file=$(subset_for_task "$task_alias")
  local run_prefix
  run_prefix=$(run_prefix_for_task "$task")
  local run_name="${run_prefix}_gatedtm_l005_t012"

  echo "============================================================"
  echo "[gpu $gpu] Starting $run_name"
  echo "============================================================"

  env     PYTHONFAULTHANDLER=1     PYTHONUNBUFFERED=1     TORCH_DISTRIBUTED_DEBUG=DETAIL     TORCH_SHOW_CPP_STACKTRACES=1     NCCL_DEBUG=INFO     NCCL_ASYNC_ERROR_HANDLING=1     MASTER_PORT="$port"     PYTHON_BIN="/home/ubuntu/anaconda3/envs/tiaf/bin/python"     SUBSET_INDICES_FILE="$subset_file"     RUN_NAME="$run_name"     TASK="$task"     VOTE_METHOD="exp"     ALPHA="5.0"     TRANSFER_SCORE="gated_temporal_margin"     TEMPORAL_LAMBDA="0.05"     TEMPORAL_TAU="0.12"     SEED="42"     BATCH_SIZE="4"     SAVE_VOTE_DEBUG="true"     GEN_LENGTH="128"     BLOCK_LENGTH="32"     TOKEN_PER_STEP="2"     bash scripts/run_retry_policy_experiment.sh "$gpu"     2>&1 | tee "logs/${run_name}.log"
}

run_queue() {
  local gpu="$1"
  local start_port="$2"
  local tasks_csv="$3"
  IFS=',' read -r -a tasks <<< "$tasks_csv"
  local offset=0
  for task in "${tasks[@]}"; do
    [ -n "$task" ] || continue
    run_candidate "$task" "$gpu" "$((start_port + offset))"
    offset=$((offset + 1))
  done
}

run_queue 0 29900 "gsm8k" &
pid0=$!
run_queue 1 29950 "svamp,countdown,math" &
pid1=$!

wait "$pid0"
wait "$pid1"

echo "============================================================"
echo "Balanced candidate run finished."
echo "label=gatedtm_l005_t012 transfer_score=gated_temporal_margin lambda=0.05 tau=0.12"
echo "gpu0_tasks=gsm8k"
echo "gpu1_tasks=svamp,countdown,math"
echo "============================================================"
