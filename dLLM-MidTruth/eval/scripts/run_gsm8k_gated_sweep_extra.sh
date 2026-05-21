#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EVAL_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$EVAL_DIR"

GPU_ID="${1:-1}"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
PYTHON_BIN="${PYTHON_BIN:-/home/work/GFlowPO/anaconda3/envs/prophet/bin/python}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SAVE_VOTE_DEBUG="${SAVE_VOTE_DEBUG:-true}"
VOTE_METHOD="${VOTE_METHOD:-exp}"
ALPHA="${ALPHA:-5.0}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29700}"

BASELINE_RUN="${BASELINE_RUN:-20260520_gsm8k_full_exp_bs4_probmargin}"
BASELINE_DIR="outputs/LLaDA-8B-Instruct/${BASELINE_RUN}"
SUBSET_FILE="analysis/transfer_score_probe/full_gsm8k_indices.txt"
mkdir -p analysis/transfer_score_probe logs
printf "%s\n" $(seq 0 1318) > "$SUBSET_FILE"

# Complementary overnight sweep:
# keep temporal signal weaker and activate it only under lower ambiguity thresholds.
CONFIGS=(
  "l008_t010:0.08:0.10"
  "l005_t010:0.05:0.10"
  "l003_t012:0.03:0.12"
  "l008_t008:0.08:0.08"
  "l005_t008:0.05:0.08"
)

extract_vote() {
  local acc_file="$1"
  grep "Vote answer" "$acc_file" | grep -oE '[0-9]+\.[0-9]+%'
}

extract_final() {
  local acc_file="$1"
  grep "Final answer" "$acc_file" | grep -oE '[0-9]+\.[0-9]+%'
}

if [ ! -f "$BASELINE_DIR/accuracy.txt" ]; then
  echo "[extra-sweep] baseline not found; running prob_margin baseline first"
  PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL \
  TORCH_SHOW_CPP_STACKTRACES=1 \
  NCCL_DEBUG=INFO \
  NCCL_ASYNC_ERROR_HANDLING=1 \
  MASTER_PORT="$MASTER_PORT_BASE" \
  PYTHON_BIN="$PYTHON_BIN" \
  SUBSET_INDICES_FILE="$SUBSET_FILE" \
  RUN_NAME="$BASELINE_RUN" \
  TASK="gsm8k" \
  VOTE_METHOD="$VOTE_METHOD" \
  ALPHA="$ALPHA" \
  TRANSFER_SCORE="prob_margin" \
  SEED="$SEED" \
  BATCH_SIZE="$BATCH_SIZE" \
  SAVE_VOTE_DEBUG="$SAVE_VOTE_DEBUG" \
    bash scripts/run_retry_policy_experiment.sh "$GPU_ID" \
    2>&1 | tee "logs/${BASELINE_RUN}.log"
fi

baseline_vote=$(extract_vote "$BASELINE_DIR/accuracy.txt")
baseline_final=$(extract_final "$BASELINE_DIR/accuracy.txt")

echo "============================================================"
echo "GSM8K gated temporal extra sweep"
echo "Baseline run: $BASELINE_RUN"
echo "Baseline vote=$baseline_vote final=$baseline_final"
echo "Order: ${CONFIGS[*]}"
echo "============================================================"

declare -a SUMMARY_ROWS
SUMMARY_ROWS+=("prob_margin|$baseline_vote|$baseline_final|baseline")

idx=0
for spec in "${CONFIGS[@]}"; do
  label="${spec%%:*}"
  rest="${spec#*:}"
  lambda="${rest%%:*}"
  tau="${rest##*:}"
  run_name="${DATE_TAG}_gsm8k_full_exp_bs4_gatedtm_${label}"
  output_dir="outputs/LLaDA-8B-Instruct/${run_name}"
  log_file="logs/${run_name}.log"

  echo "============================================================"
  echo "[extra-sweep] Running $label  (lambda=$lambda tau=$tau)"
  echo "============================================================"

  if [ ! -f "$output_dir/accuracy.txt" ]; then
    port=$((MASTER_PORT_BASE + idx + 1))
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_DISTRIBUTED_DEBUG=DETAIL \
    TORCH_SHOW_CPP_STACKTRACES=1 \
    NCCL_DEBUG=INFO \
    NCCL_ASYNC_ERROR_HANDLING=1 \
    MASTER_PORT="$port" \
    PYTHON_BIN="$PYTHON_BIN" \
    SUBSET_INDICES_FILE="$SUBSET_FILE" \
    RUN_NAME="$run_name" \
    TASK="gsm8k" \
    VOTE_METHOD="$VOTE_METHOD" \
    ALPHA="$ALPHA" \
    TRANSFER_SCORE="gated_temporal_margin" \
    TEMPORAL_LAMBDA="$lambda" \
    TEMPORAL_TAU="$tau" \
    SEED="$SEED" \
    BATCH_SIZE="$BATCH_SIZE" \
    SAVE_VOTE_DEBUG="$SAVE_VOTE_DEBUG" \
      bash scripts/run_retry_policy_experiment.sh "$GPU_ID" \
      2>&1 | tee "$log_file"
  else
    echo "[extra-sweep] existing result found, skipping run: $output_dir"
  fi

  vote=$(extract_vote "$output_dir/accuracy.txt")
  final=$(extract_final "$output_dir/accuracy.txt")
  SUMMARY_ROWS+=("gated_${label}|$vote|$final|lambda=$lambda tau=$tau")
  idx=$((idx + 1))
done

echo ""
echo "============================================================"
echo "GSM8K EXTRA GATED TEMPORAL SWEEP SUMMARY"
echo "============================================================"
for row in "${SUMMARY_ROWS[@]}"; do
  IFS='|' read -r name vote final note <<< "$row"
  printf "  %-22s vote=%-8s final=%-8s %s\n" "$name" "$vote" "$final" "$note"
done
echo "============================================================"
