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
SESSION_NAME="${SESSION_NAME:-gated_temporal_margin_suite}"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"

# Current best baseline vs C-next-1 candidate.
TRANSFER_BASELINE="${TRANSFER_BASELINE:-prob_margin}"
TRANSFER_CANDIDATE="${TRANSFER_CANDIDATE:-gated_temporal_margin}"
TEMPORAL_LAMBDA="${TEMPORAL_LAMBDA:-0.10}"
TEMPORAL_TAU="${TEMPORAL_TAU:-0.15}"

# One command, all tasks, sequential.
RUN_GSM8K="${RUN_GSM8K:-1}"
RUN_SVAMP="${RUN_SVAMP:-1}"
RUN_MATH500="${RUN_MATH500:-1}"
RUN_COUNTDOWN="${RUN_COUNTDOWN:-1}"

# Each task runs baseline on GPU A and candidate on GPU B in parallel, then waits.
GSM8K_GPU_BASE="${GSM8K_GPU_BASE:-0}"
GSM8K_GPU_CAND="${GSM8K_GPU_CAND:-1}"
SVAMP_GPU_BASE="${SVAMP_GPU_BASE:-0}"
SVAMP_GPU_CAND="${SVAMP_GPU_CAND:-1}"
MATH500_GPU_BASE="${MATH500_GPU_BASE:-0}"
MATH500_GPU_CAND="${MATH500_GPU_CAND:-1}"
COUNTDOWN_GPU_BASE="${COUNTDOWN_GPU_BASE:-0}"
COUNTDOWN_GPU_CAND="${COUNTDOWN_GPU_CAND:-1}"

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
cat > "$SUITE_SCRIPT" <<RUNNER
#!/usr/bin/env bash
set -euo pipefail
cd "$EVAL_DIR"

run_pair() {
  local task="\$1"
  local subset_file="\$2"
  local gpu_base="\$3"
  local gpu_cand="\$4"
  local port_base="\$5"
  local port_cand="\$6"
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
    MASTER_PORT="\$port_base" \
    PYTHON_BIN="$PYTHON_BIN" \
    SUBSET_INDICES_FILE="\$subset_file" \
    RUN_NAME="\${run_prefix}_probmargin" \
    TASK="\$task" \
    VOTE_METHOD="$VOTE_METHOD" \
    TRANSFER_SCORE="$TRANSFER_BASELINE" \
    SEED="$SEED" \
    BATCH_SIZE="$BATCH_SIZE" \
    SAVE_VOTE_DEBUG="$SAVE_VOTE_DEBUG" \
    bash scripts/run_retry_policy_experiment.sh "\$gpu_base" \
    2>&1 | tee "logs/\${run_prefix}_probmargin.log" &
  pid_base=\$!

  env \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_DISTRIBUTED_DEBUG=DETAIL \
    TORCH_SHOW_CPP_STACKTRACES=1 \
    NCCL_DEBUG=INFO \
    NCCL_ASYNC_ERROR_HANDLING=1 \
    MASTER_PORT="\$port_cand" \
    PYTHON_BIN="$PYTHON_BIN" \
    SUBSET_INDICES_FILE="\$subset_file" \
    RUN_NAME="\${run_prefix}_gatedtm_l${TEMPORAL_LAMBDA//./}_t${TEMPORAL_TAU//./}" \
    TASK="\$task" \
    VOTE_METHOD="$VOTE_METHOD" \
    TRANSFER_SCORE="$TRANSFER_CANDIDATE" \
    TEMPORAL_LAMBDA="$TEMPORAL_LAMBDA" \
    TEMPORAL_TAU="$TEMPORAL_TAU" \
    SEED="$SEED" \
    BATCH_SIZE="$BATCH_SIZE" \
    SAVE_VOTE_DEBUG="$SAVE_VOTE_DEBUG" \
    bash scripts/run_retry_policy_experiment.sh "\$gpu_cand" \
    2>&1 | tee "logs/\${run_prefix}_gatedtm_l${TEMPORAL_LAMBDA//./}_t${TEMPORAL_TAU//./}.log" &
  pid_cand=\$!

  wait "\$pid_base"
  wait "\$pid_cand"

  echo "Finished task: \$task"
}

RUNNER

if [ "$RUN_GSM8K" = "1" ]; then
  cat >> "$SUITE_SCRIPT" <<RUNNER
run_pair "gsm8k" \
  "$EVAL_DIR/analysis/transfer_score_probe/full_gsm8k_indices.txt" \
  "$GSM8K_GPU_BASE" "$GSM8K_GPU_CAND" \
  29515 29516 \
  "${DATE_TAG}_gsm8k_full_exp_bs4"

RUNNER
fi

if [ "$RUN_SVAMP" = "1" ]; then
  cat >> "$SUITE_SCRIPT" <<RUNNER
run_pair "svamp" \
  "$EVAL_DIR/analysis/transfer_score_probe/full_svamp_indices.txt" \
  "$SVAMP_GPU_BASE" "$SVAMP_GPU_CAND" \
  29517 29518 \
  "${DATE_TAG}_svamp_full_exp_bs4"

RUNNER
fi

if [ "$RUN_MATH500" = "1" ]; then
  cat >> "$SUITE_SCRIPT" <<RUNNER
run_pair "math" \
  "$EVAL_DIR/analysis/transfer_score_probe/full_math500_indices.txt" \
  "$MATH500_GPU_BASE" "$MATH500_GPU_CAND" \
  29519 29520 \
  "${DATE_TAG}_math500_full_exp_bs4"

RUNNER
fi

if [ "$RUN_COUNTDOWN" = "1" ]; then
  cat >> "$SUITE_SCRIPT" <<RUNNER
run_pair "countdown" \
  "$EVAL_DIR/analysis/transfer_score_probe/full_countdown_indices.txt" \
  "$COUNTDOWN_GPU_BASE" "$COUNTDOWN_GPU_CAND" \
  29521 29522 \
  "${DATE_TAG}_countdown_full_exp_bs4"

RUNNER
fi

cat >> "$SUITE_SCRIPT" <<'RUNNER'
echo "============================================================"
echo "All requested tasks finished."
echo "============================================================"
RUNNER

chmod +x "$SUITE_SCRIPT"

if command -v tmux >/dev/null 2>&1; then
  tmux new-session -d -s "$SESSION_NAME" "bash '$SUITE_SCRIPT'"
  echo "Launched sequential suite in tmux session: $SESSION_NAME"
  echo "Attach with: tmux attach -t $SESSION_NAME"
else
  NOHUP_LOG="$EVAL_DIR/logs/${SESSION_NAME}.nohup.log"
  PID_FILE="$EVAL_DIR/logs/${SESSION_NAME}.pid"
  nohup bash "$SUITE_SCRIPT" > "$NOHUP_LOG" 2>&1 < /dev/null &
  echo $! > "$PID_FILE"
  echo "tmux not found; launched sequential suite with nohup instead."
  echo "PID file: $PID_FILE"
  echo "Nohup log: $NOHUP_LOG"
fi

echo "Monitor logs under: $EVAL_DIR/logs/"
echo "Baseline:  $TRANSFER_BASELINE"
echo "Candidate: $TRANSFER_CANDIDATE (lambda=$TEMPORAL_LAMBDA tau=$TEMPORAL_TAU)"
