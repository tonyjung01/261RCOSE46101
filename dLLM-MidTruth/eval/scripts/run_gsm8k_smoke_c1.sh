#!/usr/bin/env bash
# GSM8K smoke test for token-ordering C1 (temporal_margin).
#
# Runs five conditions sequentially on a single GPU:
#   A  : top1_prob
#   B  : prob_margin
#   C1a: temporal_margin, lambda=0.05
#   C1b: temporal_margin, lambda=0.10
#   C1c: temporal_margin, lambda=0.20
#
# Usage:
#   bash scripts/run_gsm8k_smoke_c1.sh [GPU_ID]
#
# Optional env overrides:
#   SMOKE_N=64          number of GSM8K samples (default: 64)
#   SMOKE_SEED=42       random seed (default: 42)
#   MASTER_PORT=29500   torchrun port (default: 29500)
#   DATE_TAG=20260519   date prefix for output dirs (default: today)
#
# IMPORTANT — batch size parity:
#   This script defaults to BATCH_SIZE=4 to match the established A/B baseline
#   (20260515_*_bs4_*). Any standalone call to run_retry_policy_experiment.sh
#   must also set BATCH_SIZE=4 explicitly; the runner's own default is 8.
#
# Standalone C1-only example (apples-to-apples with A/B bs4 baseline):
#   TRANSFER_SCORE=temporal_margin TEMPORAL_LAMBDA=0.10 BATCH_SIZE=4 \
#   SUBSET_INDICES_FILE=analysis/transfer_score_probe/smoke_gsm8k_n64_seed42.txt \
#   RUN_NAME=20260519_gsm8k_smoke_C1b_temporal_l010 \
#   TASK=gsm8k VOTE_METHOD=exp ALPHA=5.0 SAVE_VOTE_DEBUG=true \
#   bash scripts/run_retry_policy_experiment.sh 0

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EVAL_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$EVAL_DIR"

GPU_ID="${1:-0}"
SMOKE_N="${SMOKE_N:-64}"
SMOKE_SEED="${SMOKE_SEED:-42}"
MASTER_PORT="${MASTER_PORT:-29500}"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"

PYTHON_BIN="${PYTHON_BIN:-/home/work/GFlowPO/anaconda3/envs/prophet/bin/python}"
VOTE_METHOD="${VOTE_METHOD:-exp}"
ALPHA="${ALPHA:-5.0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SAVE_VOTE_DEBUG="${SAVE_VOTE_DEBUG:-true}"

# ── Build smoke subset index file ────────────────────────────────────────────
SMOKE_DIR="analysis/transfer_score_probe"
SUBSET_FILE="$SMOKE_DIR/smoke_gsm8k_n${SMOKE_N}_seed${SMOKE_SEED}.txt"
mkdir -p "$SMOKE_DIR" logs

if [ ! -f "$SUBSET_FILE" ]; then
  "$PYTHON_BIN" - "$SMOKE_N" "$SMOKE_SEED" "$SUBSET_FILE" <<'PY'
import sys, random
n, seed, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
random.seed(seed)
indices = random.sample(range(1319), n)
with open(out, "w") as f:
    f.write("\n".join(str(i) for i in sorted(indices)) + "\n")
print(f"[smoke] wrote {n} indices → {out}")
PY
fi

# ── Helper ────────────────────────────────────────────────────────────────────
run_condition() {
  local label="$1"
  local transfer_score="$2"
  local temporal_lambda="${3:-0.0}"
  local run_name="${DATE_TAG}_gsm8k_smoke_${label}"
  local output_dir="outputs/LLaDA-8B-Instruct/${run_name}"

  if [ -f "$output_dir/rank_0_generations.json" ]; then
    echo "[smoke] $label already done — skipping ($output_dir)"
    return 0
  fi

  echo "============================================================"
  echo "[smoke] Running: $label  (transfer_score=$transfer_score  lambda=$temporal_lambda)"
  echo "============================================================"

  PYTHON_BIN="$PYTHON_BIN" \
  SUBSET_INDICES_FILE="$SUBSET_FILE" \
  RUN_NAME="$run_name" \
  TASK="gsm8k" \
  VOTE_METHOD="$VOTE_METHOD" \
  ALPHA="$ALPHA" \
  TRANSFER_SCORE="$transfer_score" \
  TEMPORAL_LAMBDA="$temporal_lambda" \
  SEED="$SMOKE_SEED" \
  BATCH_SIZE="$BATCH_SIZE" \
  SAVE_VOTE_DEBUG="$SAVE_VOTE_DEBUG" \
  MASTER_PORT="$MASTER_PORT" \
  OUTPUT_DIR="$output_dir" \
    bash scripts/run_retry_policy_experiment.sh "$GPU_ID" \
    2>&1 | tee "logs/${run_name}.log"

  echo "[smoke] $label done → $output_dir/accuracy.txt"
}

# ── Run all five conditions ───────────────────────────────────────────────────
run_condition "A_top1prob"         "top1_prob"       "0.0"
run_condition "B_probmargin"       "prob_margin"     "0.0"
run_condition "C1a_temporal_l005" "temporal_margin" "0.05"
run_condition "C1b_temporal_l010" "temporal_margin" "0.10"
run_condition "C1c_temporal_l020" "temporal_margin" "0.20"

# ── Print summary ─────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "SMOKE SUMMARY  (vote_answer accuracy — primary metric)"
echo "============================================================"
for label in A_top1prob B_probmargin C1a_temporal_l005 C1b_temporal_l010 C1c_temporal_l020; do
  acc_file="outputs/LLaDA-8B-Instruct/${DATE_TAG}_gsm8k_smoke_${label}/accuracy.txt"
  if [ -f "$acc_file" ]; then
    # grep "Vote answer" explicitly to avoid reading Final answer accuracy (printed first in get_acc.py)
    vote_pct=$(grep "Vote answer" "$acc_file" | grep -oE '[0-9]+\.[0-9]+%' || true)
    final_pct=$(grep "Final answer" "$acc_file" | grep -oE '[0-9]+\.[0-9]+%' || true)
    printf "  %-30s vote=%s  final=%s\n" "$label" "${vote_pct:-(n/a)}" "${final_pct:-(n/a)}"
  else
    printf "  %-30s %s\n" "$label" "(no accuracy.txt)"
  fi
done
echo "============================================================"
echo "Full logs: $EVAL_DIR/logs/"
echo "Outputs:   $EVAL_DIR/outputs/LLaDA-8B-Instruct/"
