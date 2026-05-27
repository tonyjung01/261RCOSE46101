#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EVAL_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$EVAL_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/work/GFlowPO/anaconda3/envs/prophet/bin/python}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SAVE_VOTE_DEBUG="${SAVE_VOTE_DEBUG:-true}"
VOTE_METHOD="${VOTE_METHOD:-exp}"
SESSION_NAME="${SESSION_NAME:-transfer_score_suite}"

# Defaults now match the requested "one command, all tasks, sequential" behavior.
RUN_GSM8K="${RUN_GSM8K:-1}"
RUN_SVAMP="${RUN_SVAMP:-1}"
RUN_MATH500="${RUN_MATH500:-1}"
RUN_COUNTDOWN="${RUN_COUNTDOWN:-1}"

# GPU pairing per task. Each task runs top1_prob on GPU A and prob_margin on GPU B in parallel,
# then waits before moving on to the next task.
GSM8K_GPU_TOP1="${GSM8K_GPU_TOP1:-0}"
GSM8K_GPU_MARGIN="${GSM8K_GPU_MARGIN:-1}"
SVAMP_GPU_TOP1="${SVAMP_GPU_TOP1:-0}"
SVAMP_GPU_MARGIN="${SVAMP_GPU_MARGIN:-1}"
MATH500_GPU_TOP1="${MATH500_GPU_TOP1:-0}"
MATH500_GPU_MARGIN="${MATH500_GPU_MARGIN:-1}"
COUNTDOWN_GPU_TOP1="${COUNTDOWN_GPU_TOP1:-0}"
COUNTDOWN_GPU_MARGIN="${COUNTDOWN_GPU_MARGIN:-1}"

mkdir -p analysis/transfer_score_probe logs

printf "%s\n" $(seq 0 1318) > analysis/transfer_score_probe/full_gsm8k_indices.txt
printf "%s\n" $(seq 0 299) > analysis/transfer_score_probe/full_svamp_indices.txt
printf "%s\n" $(seq 0 499) > analysis/transfer_score_probe/full_math500_indices.txt
printf "%s\n" $(seq 0 255) > analysis/transfer_score_probe/full_countdown_indices.txt

echo "Running dataset preflight checks"
"$PYTHON_BIN" - <<'PY'
from datasets import load_dataset
checks = [
    ("gsm8k", lambda: load_dataset("openai/gsm8k", "main", split="test")),
    ("svamp", lambda: load_dataset("ChilleD/SVAMP", "default", split="test")),
    ("math500", lambda: load_dataset("HuggingFaceH4/MATH-500", split="test")),
]
for name, fn in checks:
    ds = fn()
    print(name, "OK", len(ds))
PY

COUNTDOWN_FILE="$EVAL_DIR/../dataset/countdown_cd3_test.jsonl"
if [ ! -f "$COUNTDOWN_FILE" ]; then
  echo "Countdown dataset file missing: $COUNTDOWN_FILE"
  exit 1
fi
echo "countdown OK $(wc -l < "$COUNTDOWN_FILE")"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME"
  echo "Kill it first with: tmux kill-session -t $SESSION_NAME"
  exit 1
fi

SUITE_SCRIPT="$EVAL_DIR/logs/${SESSION_NAME}_runner.sh"
cat > "$SUITE_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$EVAL_DIR"

run_pair() {
  local task="\$1"
  local subset_file="\$2"
  local gpu_top1="\$3"
  local gpu_margin="\$4"
  local port_top1="\$5"
  local port_margin="\$6"
  local run_prefix="\$7"

  echo "============================================================"
  echo "Starting task: \$task"
  echo "============================================================"

  env \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_DISTRIBUTED_DEBUG=DETAIL \
    TORCH_SHOW_CPP_STACKTRACES=1 \
    NCCL_DEBUG=INFO \
    NCCL_ASYNC_ERROR_HANDLING=1 \
    MASTER_PORT="\$port_top1" \
    PYTHON_BIN="$PYTHON_BIN" \
    SUBSET_INDICES_FILE="\$subset_file" \
    RUN_NAME="\${run_prefix}_top1prob" \
    TASK="\$task" \
    VOTE_METHOD="$VOTE_METHOD" \
    TRANSFER_SCORE="top1_prob" \
    SEED="$SEED" \
    BATCH_SIZE="$BATCH_SIZE" \
    SAVE_VOTE_DEBUG="$SAVE_VOTE_DEBUG" \
    bash scripts/run_retry_policy_experiment.sh "\$gpu_top1" \
    2>&1 | tee "logs/\${run_prefix}_top1prob.log" &
  pid_top1=\$!

  env \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_DISTRIBUTED_DEBUG=DETAIL \
    TORCH_SHOW_CPP_STACKTRACES=1 \
    NCCL_DEBUG=INFO \
    NCCL_ASYNC_ERROR_HANDLING=1 \
    MASTER_PORT="\$port_margin" \
    PYTHON_BIN="$PYTHON_BIN" \
    SUBSET_INDICES_FILE="\$subset_file" \
    RUN_NAME="\${run_prefix}_probmargin" \
    TASK="\$task" \
    VOTE_METHOD="$VOTE_METHOD" \
    TRANSFER_SCORE="prob_margin" \
    SEED="$SEED" \
    BATCH_SIZE="$BATCH_SIZE" \
    SAVE_VOTE_DEBUG="$SAVE_VOTE_DEBUG" \
    bash scripts/run_retry_policy_experiment.sh "\$gpu_margin" \
    2>&1 | tee "logs/\${run_prefix}_probmargin.log" &
  pid_margin=\$!

  wait "\$pid_top1"
  wait "\$pid_margin"

  echo "Finished task: \$task"
}

EOF

if [ "$RUN_GSM8K" = "1" ]; then
  cat >> "$SUITE_SCRIPT" <<EOF
run_pair "gsm8k" \
  "$EVAL_DIR/analysis/transfer_score_probe/full_gsm8k_indices.txt" \
  "$GSM8K_GPU_TOP1" "$GSM8K_GPU_MARGIN" \
  29415 29416 \
  "20260515_gsm8k_full_exp_bs4"

EOF
fi

if [ "$RUN_SVAMP" = "1" ]; then
  cat >> "$SUITE_SCRIPT" <<EOF
run_pair "svamp" \
  "$EVAL_DIR/analysis/transfer_score_probe/full_svamp_indices.txt" \
  "$SVAMP_GPU_TOP1" "$SVAMP_GPU_MARGIN" \
  29417 29418 \
  "20260515_svamp_full_exp_bs4"

EOF
fi

if [ "$RUN_MATH500" = "1" ]; then
  cat >> "$SUITE_SCRIPT" <<EOF
run_pair "math" \
  "$EVAL_DIR/analysis/transfer_score_probe/full_math500_indices.txt" \
  "$MATH500_GPU_TOP1" "$MATH500_GPU_MARGIN" \
  29419 29420 \
  "20260515_math500_full_exp_bs4"

EOF
fi

if [ "$RUN_COUNTDOWN" = "1" ]; then
  cat >> "$SUITE_SCRIPT" <<EOF
run_pair "countdown" \
  "$EVAL_DIR/analysis/transfer_score_probe/full_countdown_indices.txt" \
  "$COUNTDOWN_GPU_TOP1" "$COUNTDOWN_GPU_MARGIN" \
  29421 29422 \
  "20260515_countdown_full_exp_bs4"

EOF
fi

cat >> "$SUITE_SCRIPT" <<'EOF'
echo "============================================================"
echo "All requested tasks finished."
echo "============================================================"
EOF

chmod +x "$SUITE_SCRIPT"

tmux new-session -d -s "$SESSION_NAME" "bash '$SUITE_SCRIPT'"

echo "Launched sequential suite in tmux session: $SESSION_NAME"
echo "Attach with: tmux attach -t $SESSION_NAME"
echo "Monitor current task logs under: $EVAL_DIR/logs/"
