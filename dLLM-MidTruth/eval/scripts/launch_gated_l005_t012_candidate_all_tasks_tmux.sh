#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EVAL_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$EVAL_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/work/GFlowPO/anaconda3/envs/prophet/bin/python}"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
SESSION_NAME="${SESSION_NAME:-gated_l005_t012_candidate_all_tasks}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SAVE_VOTE_DEBUG="${SAVE_VOTE_DEBUG:-true}"
VOTE_METHOD="${VOTE_METHOD:-exp}"
ALPHA="${ALPHA:-5.0}"

TEMPORAL_LAMBDA="${TEMPORAL_LAMBDA:-0.05}"
TEMPORAL_TAU="${TEMPORAL_TAU:-0.12}"
LABEL="${LABEL:-gatedtm_l005_t012}"

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

run_candidate() {
  local task="\$1"
  local subset_file="\$2"
  local gpu="\$3"
  local port="\$4"
  local run_prefix="\$5"
  local run_name="\${run_prefix}_${LABEL}"

  echo "============================================================"
  echo "Starting candidate: \$run_name on GPU \$gpu"
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
    TRANSFER_SCORE="gated_temporal_margin" \
    TEMPORAL_LAMBDA="$TEMPORAL_LAMBDA" \
    TEMPORAL_TAU="$TEMPORAL_TAU" \
    SEED="$SEED" \
    BATCH_SIZE="$BATCH_SIZE" \
    SAVE_VOTE_DEBUG="$SAVE_VOTE_DEBUG" \
    bash scripts/run_retry_policy_experiment.sh "\$gpu" \
    2>&1 | tee "logs/\${run_name}.log"
}

run_candidate "gsm8k" "$EVAL_DIR/analysis/transfer_score_probe/full_gsm8k_indices.txt" 0 29815 "${DATE_TAG}_gsm8k_full_exp_bs4" &
pid_gsm=\$!
run_candidate "svamp" "$EVAL_DIR/analysis/transfer_score_probe/full_svamp_indices.txt" 1 29816 "${DATE_TAG}_svamp_full_exp_bs4" &
pid_svamp=\$!
wait "\$pid_gsm"
wait "\$pid_svamp"

run_candidate "math" "$EVAL_DIR/analysis/transfer_score_probe/full_math500_indices.txt" 0 29817 "${DATE_TAG}_math500_full_exp_bs4" &
pid_math=\$!
run_candidate "countdown" "$EVAL_DIR/analysis/transfer_score_probe/full_countdown_indices.txt" 1 29818 "${DATE_TAG}_countdown_full_exp_bs4" &
pid_count=\$!
wait "\$pid_math"
wait "\$pid_count"

echo "============================================================"
echo "Candidate all-task run finished: $LABEL"
echo "lambda=$TEMPORAL_LAMBDA tau=$TEMPORAL_TAU"
echo "============================================================"
RUNNER

chmod +x "$RUNNER"
tmux new-session -d -s "$SESSION_NAME" "bash '$EVAL_DIR/$RUNNER'"

echo "Launched candidate-only all-task run in tmux: $SESSION_NAME"
echo "Candidate: gated_temporal_margin lambda=$TEMPORAL_LAMBDA tau=$TEMPORAL_TAU"
echo "Attach with: tmux attach -t $SESSION_NAME"
