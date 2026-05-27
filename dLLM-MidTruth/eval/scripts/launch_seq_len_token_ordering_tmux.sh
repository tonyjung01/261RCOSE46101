#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EVAL_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$EVAL_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/work/GFlowPO/anaconda3/envs/prophet/bin/python}"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
SESSION_NAME="${SESSION_NAME:-seq_len_token_ordering_sweep}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SAVE_VOTE_DEBUG="${SAVE_VOTE_DEBUG:-true}"
VOTE_METHOD="${VOTE_METHOD:-exp}"
ALPHA="${ALPHA:-5.0}"

# Length-generalization sweep:
#   methods: top1_prob, prob_margin, gated_temporal_margin
#   lengths: 256, 512
#   tasks  : gsm8k, svamp, math, countdown
#
# Default gated setting is the GSM8K-best setting from the 128-length sweep.
# Override with GATED_LAMBDA=0.10 GATED_TAU=0.15 for the global 4/4 setting.
GATED_LAMBDA="${GATED_LAMBDA:-0.05}"
GATED_TAU="${GATED_TAU:-0.15}"
GATED_LABEL="${GATED_LABEL:-gatedtm_l005_t015}"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPU_FREE_MEM_MB="${GPU_FREE_MEM_MB:-2000}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-60}"

mkdir -p analysis/transfer_score_probe logs
printf "%s\n" $(seq 0 1318) > analysis/transfer_score_probe/full_gsm8k_indices.txt
printf "%s\n" $(seq 0 299) > analysis/transfer_score_probe/full_svamp_indices.txt
printf "%s\n" $(seq 0 499) > analysis/transfer_score_probe/full_math500_indices.txt
printf "%s\n" $(seq 0 255) > analysis/transfer_score_probe/full_countdown_indices.txt

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME"
  exit 1
fi

RUNNER="logs/${SESSION_NAME}_runner.sh"
cat > "$RUNNER" <<RUNNER
#!/usr/bin/env bash
set -euo pipefail
cd "$EVAL_DIR"

wait_for_gpu_free() {
  local gpu="\$1"
  while true; do
    local used
    used=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "\$gpu" | tr -d ' ')
    if [ "\$used" -lt "$GPU_FREE_MEM_MB" ]; then
      echo "[scheduler] GPU \$gpu is free enough: \${used}MiB used"
      return 0
    fi
    echo "[scheduler] GPU \$gpu busy: \${used}MiB used; sleeping ${GPU_WAIT_SECONDS}s"
    sleep "$GPU_WAIT_SECONDS"
  done
}

subset_for_task() {
  case "\$1" in
    gsm8k) echo "$EVAL_DIR/analysis/transfer_score_probe/full_gsm8k_indices.txt" ;;
    svamp) echo "$EVAL_DIR/analysis/transfer_score_probe/full_svamp_indices.txt" ;;
    math|math500) echo "$EVAL_DIR/analysis/transfer_score_probe/full_math500_indices.txt" ;;
    countdown) echo "$EVAL_DIR/analysis/transfer_score_probe/full_countdown_indices.txt" ;;
    *) echo "Unknown task: \$1" >&2; return 1 ;;
  esac
}

method_to_transfer_score() {
  case "\$1" in
    top1) echo "top1_prob" ;;
    margin) echo "prob_margin" ;;
    gated) echo "gated_temporal_margin" ;;
    *) echo "Unknown method: \$1" >&2; return 1 ;;
  esac
}

method_to_label() {
  case "\$1" in
    top1) echo "top1prob" ;;
    margin) echo "probmargin" ;;
    gated) echo "$GATED_LABEL" ;;
    *) echo "unknown" ;;
  esac
}

run_job() {
  local gen_length="\$1"
  local method="\$2"
  local task="\$3"
  local gpu="\$4"
  local port="\$5"
  local transfer_score
  transfer_score=\$(method_to_transfer_score "\$method")
  local method_label
  method_label=\$(method_to_label "\$method")
  local task_label="\$task"
  if [ "\$task" = "math" ]; then
    task_label="math500"
  fi
  local subset_file
  subset_file=\$(subset_for_task "\$task")
  local run_name="${DATE_TAG}_\${task_label}_full_exp_bs${BATCH_SIZE}_len\${gen_length}_\${method_label}"

  echo "============================================================"
  echo "[gpu \$gpu] Starting \$run_name"
  echo "  gen_length=\$gen_length transfer_score=\$transfer_score"
  echo "============================================================"

  local -a extra_env=()
  if [ "\$transfer_score" = "gated_temporal_margin" ]; then
    extra_env+=(TEMPORAL_LAMBDA="$GATED_LAMBDA" TEMPORAL_TAU="$GATED_TAU")
  fi

  env \\
    PYTHONFAULTHANDLER=1 \\
    PYTHONUNBUFFERED=1 \\
    TORCH_DISTRIBUTED_DEBUG=DETAIL \\
    TORCH_SHOW_CPP_STACKTRACES=1 \\
    NCCL_DEBUG=INFO \\
    NCCL_ASYNC_ERROR_HANDLING=1 \\
    MASTER_PORT="\$port" \\
    PYTHON_BIN="$PYTHON_BIN" \\
    SUBSET_INDICES_FILE="\$subset_file" \\
    RUN_NAME="\$run_name" \\
    TASK="\$task" \\
    VOTE_METHOD="$VOTE_METHOD" \\
    ALPHA="$ALPHA" \\
    TRANSFER_SCORE="\$transfer_score" \\
    GEN_LENGTH="\$gen_length" \\
    SEED="$SEED" \\
    BATCH_SIZE="$BATCH_SIZE" \\
    SAVE_VOTE_DEBUG="$SAVE_VOTE_DEBUG" \\
    "\${extra_env[@]}" \\
    bash scripts/run_retry_policy_experiment.sh "\$gpu" \\
    2>&1 | tee "logs/\${run_name}.log"
}

run_queue() {
  local gpu="\$1"
  local start_port="\$2"
  shift 2
  wait_for_gpu_free "\$gpu"
  local offset=0
  while [ "\$#" -gt 0 ]; do
    run_job "\$1" "\$2" "\$3" "\$gpu" "\$((start_port + offset))"
    shift 3
    offset=\$((offset + 1))
  done
}

# Balanced by approximate cost: num_samples * gen_length.
# GPU0: 512 GSM8K plus 256 short tasks.
# GPU1: 256 GSM8K plus 512 short tasks.
run_queue "$GPU1" 30100 \\
  256 top1 gsm8k      256 margin gsm8k      256 gated gsm8k \\
  512 top1 countdown  512 margin countdown  512 gated countdown \\
  512 top1 math       512 margin math       512 gated math \\
  512 top1 svamp      512 margin svamp      512 gated svamp &
pid1=\$!

run_queue "$GPU0" 30200 \\
  512 top1 gsm8k      512 margin gsm8k      512 gated gsm8k \\
  256 top1 svamp      256 margin svamp      256 gated svamp \\
  256 top1 math       256 margin math       256 gated math \\
  256 top1 countdown  256 margin countdown  256 gated countdown &
pid0=\$!

wait "\$pid1"
wait "\$pid0"

echo "============================================================"
echo "Sequence-length token-ordering sweep finished."
echo "lengths=256,512 methods=top1_prob,prob_margin,gated_temporal_margin"
echo "gated_lambda=$GATED_LAMBDA gated_tau=$GATED_TAU gated_label=$GATED_LABEL"
echo "============================================================"
RUNNER

chmod +x "$RUNNER"
tmux new-session -d -s "$SESSION_NAME" "bash '$EVAL_DIR/$RUNNER'"

echo "Launched sequence-length token-ordering sweep in tmux: $SESSION_NAME"
echo "Methods: top1_prob, prob_margin, gated_temporal_margin"
echo "Lengths: 256, 512"
echo "Gated: lambda=$GATED_LAMBDA tau=$GATED_TAU label=$GATED_LABEL"
echo "GPU1 starts immediately if free; GPU0 waits until memory.used < ${GPU_FREE_MEM_MB}MiB."
echo "Attach with: tmux attach -t $SESSION_NAME"
