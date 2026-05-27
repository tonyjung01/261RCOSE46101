#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EVAL_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$EVAL_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/anaconda3/envs/tiaf/bin/python}"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
SESSION_NAME="${SESSION_NAME:-candidate_all_tasks_balanced}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SAVE_VOTE_DEBUG="${SAVE_VOTE_DEBUG:-true}"
VOTE_METHOD="${VOTE_METHOD:-exp}"
ALPHA="${ALPHA:-5.0}"

GEN_LENGTH="${GEN_LENGTH:-128}"
BLOCK_LENGTH="${BLOCK_LENGTH:-32}"
TOKEN_PER_STEP="${TOKEN_PER_STEP:-2}"

TRANSFER_SCORE="${TRANSFER_SCORE:-gated_temporal_margin}"
TEMPORAL_LAMBDA="${TEMPORAL_LAMBDA:-0.05}"
TEMPORAL_TAU="${TEMPORAL_TAU:-0.12}"
LABEL="${LABEL:-gatedtm_l005_t012}"

# Balanced default:
# GPU0 handles the longest task alone.
# GPU1 drains the shorter tasks sequentially, so it does not wait for GSM8K.
GPU0_TASKS="${GPU0_TASKS:-gsm8k}"
GPU1_TASKS="${GPU1_TASKS:-svamp,countdown,math}"

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

subset_for_task() {
  case "\$1" in
    gsm8k) echo "$EVAL_DIR/analysis/transfer_score_probe/full_gsm8k_indices.txt" ;;
    svamp) echo "$EVAL_DIR/analysis/transfer_score_probe/full_svamp_indices.txt" ;;
    math|math500) echo "$EVAL_DIR/analysis/transfer_score_probe/full_math500_indices.txt" ;;
    countdown) echo "$EVAL_DIR/analysis/transfer_score_probe/full_countdown_indices.txt" ;;
    *) echo "Unknown task: \$1" >&2; return 1 ;;
  esac
}

eval_task_name() {
  case "\$1" in
    math500) echo "math" ;;
    *) echo "\$1" ;;
  esac
}

run_prefix_for_task() {
  case "\$1" in
    math) echo "${DATE_TAG}_math500_full_exp_bs4" ;;
    *) echo "${DATE_TAG}_\${1}_full_exp_bs4" ;;
  esac
}

run_candidate() {
  local task_alias="\$1"
  local gpu="\$2"
  local port="\$3"
  local task
  task=\$(eval_task_name "\$task_alias")
  local subset_file
  subset_file=\$(subset_for_task "\$task_alias")
  local run_prefix
  run_prefix=\$(run_prefix_for_task "\$task")
  local run_name="\${run_prefix}_${LABEL}"

  echo "============================================================"
  echo "[gpu \$gpu] Starting \$run_name"
  echo "============================================================"

  env \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_DISTRIBUTED_DEBUG=DETAIL \
    TORCH_SHOW_CPP_STACKTRACES=1 \
    NCCL_DEBUG=INFO \
    NCCL_ASYNC_ERROR_HANDLING=1 \
    MASTER_PORT="\$port" \
    PYTHON_BIN="$PYTHON_BIN" \
    SUBSET_INDICES_FILE="\$subset_file" \
    RUN_NAME="\$run_name" \
    TASK="\$task" \
    VOTE_METHOD="$VOTE_METHOD" \
    ALPHA="$ALPHA" \
    TRANSFER_SCORE="$TRANSFER_SCORE" \
    TEMPORAL_LAMBDA="$TEMPORAL_LAMBDA" \
    TEMPORAL_TAU="$TEMPORAL_TAU" \
    SEED="$SEED" \
    BATCH_SIZE="$BATCH_SIZE" \
    SAVE_VOTE_DEBUG="$SAVE_VOTE_DEBUG" \
    GEN_LENGTH="$GEN_LENGTH" \
    BLOCK_LENGTH="$BLOCK_LENGTH" \
    TOKEN_PER_STEP="$TOKEN_PER_STEP" \
    bash scripts/run_retry_policy_experiment.sh "\$gpu" \
    2>&1 | tee "logs/\${run_name}.log"
}

run_queue() {
  local gpu="\$1"
  local start_port="\$2"
  local tasks_csv="\$3"
  IFS=',' read -r -a tasks <<< "\$tasks_csv"
  local offset=0
  for task in "\${tasks[@]}"; do
    [ -n "\$task" ] || continue
    run_candidate "\$task" "\$gpu" "\$((start_port + offset))"
    offset=\$((offset + 1))
  done
}

run_queue 0 29900 "$GPU0_TASKS" &
pid0=\$!
run_queue 1 29950 "$GPU1_TASKS" &
pid1=\$!

wait "\$pid0"
wait "\$pid1"

echo "============================================================"
echo "Balanced candidate run finished."
echo "label=$LABEL transfer_score=$TRANSFER_SCORE lambda=$TEMPORAL_LAMBDA tau=$TEMPORAL_TAU"
echo "gpu0_tasks=$GPU0_TASKS"
echo "gpu1_tasks=$GPU1_TASKS"
echo "============================================================"
RUNNER

chmod +x "$RUNNER"
tmux new-session -d -s "$SESSION_NAME" "bash '$EVAL_DIR/$RUNNER'"

echo "Launched balanced candidate run in tmux: $SESSION_NAME"
echo "GPU0 tasks: $GPU0_TASKS"
echo "GPU1 tasks: $GPU1_TASKS"
echo "Candidate: $TRANSFER_SCORE label=$LABEL lambda=$TEMPORAL_LAMBDA tau=$TEMPORAL_TAU"
echo "Attach with: tmux attach -t $SESSION_NAME"
