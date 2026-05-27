#!/usr/bin/env bash
# Sweep new ordering formulas with LLaDA-8B-Instruct, seq_len=128.
# Formulas: top1_x_margin, top1_heavy_blend, margin_heavy_blend,
#            margin_exp_kl (gamma in {1,2,5})
# Baseline: prob_margin (reference for comparison)
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EVAL_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$EVAL_DIR"

GPU_ARGS=("${@:-0}")
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/anaconda3/envs/tiaf/bin/python}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-4}"
VOTE_METHOD="${VOTE_METHOD:-exp}"
ALPHA="${ALPHA:-5.0}"
GEN_LENGTH="${GEN_LENGTH:-128}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29700}"

MODEL_PATH="${MODEL_PATH:-/home/ubuntu/pbm/261RCOSE46101/dLLM-MidTruth/pretrained_weights/GSAI-ML/LLaDA-8B-Instruct}"
MODEL_NAME="${MODEL_NAME:-LLaDA-8B-Instruct}"

SUBSET_FILE="analysis/transfer_score_probe/full_gsm8k_indices.txt"
mkdir -p logs

if [ ! -f "$SUBSET_FILE" ]; then
  printf "%s\n" $(seq 0 1318) > "$SUBSET_FILE"
fi

# Each entry: "run_label:TRANSFER_SCORE[:KL_GAMMA]"
CONFIGS=(
  "probmargin:prob_margin"
  "top1xmargin:top1_x_margin"
  "top1heavyblend:top1_heavy_blend"
  "marginheavyblend:margin_heavy_blend"
  "margexkl_g1:margin_exp_kl:1"
  "margexkl_g2:margin_exp_kl:2"
  "margexkl_g5:margin_exp_kl:5"
)

port=$MASTER_PORT_BASE
for spec in "${CONFIGS[@]}"; do
  IFS=':' read -r label tscore kl_gamma_val <<< "$spec"
  kl_gamma_val="${kl_gamma_val:-1.0}"

  run_name="${DATE_TAG}_gsm8k_8b_g${GEN_LENGTH}_${label}_seed${SEED}"
  output_dir="outputs/${MODEL_NAME}/${run_name}"
  log_file="logs/${run_name}.log"

  if [ -f "$output_dir/accuracy.txt" ]; then
    echo "[sweep] skip (already done): $run_name"
    port=$((port + 1))
    continue
  fi

  echo "============================================================"
  echo "[sweep] $label  transfer_score=$tscore kl_gamma=$kl_gamma_val"
  echo "============================================================"

  PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  MASTER_PORT="$port" \
  PYTHON_BIN="$PYTHON_BIN" \
  MODEL_PATH="$MODEL_PATH" \
  MODEL_NAME="$MODEL_NAME" \
  SUBSET_INDICES_FILE="$SUBSET_FILE" \
  RUN_NAME="$run_name" \
  TASK="gsm8k" \
  VOTE_METHOD="$VOTE_METHOD" \
  ALPHA="$ALPHA" \
  GEN_LENGTH="$GEN_LENGTH" \
  TRANSFER_SCORE="$tscore" \
  KL_GAMMA="$kl_gamma_val" \
  SEED="$SEED" \
  BATCH_SIZE="$BATCH_SIZE" \
  SAVE_VOTE_DEBUG="true" \
    bash scripts/run_retry_policy_experiment.sh "${GPU_ARGS[@]}" \
    2>&1 | tee "$log_file"

  port=$((port + 1))
done

echo ""
echo "============================================================"
echo "ORDERING SWEEP SUMMARY  (${DATE_TAG}, 8B-Instruct, gen_length=${GEN_LENGTH})"
echo "============================================================"
for spec in "${CONFIGS[@]}"; do
  IFS=':' read -r label tscore kl_gamma_val <<< "$spec"
  kl_gamma_val="${kl_gamma_val:-1.0}"
  run_name="${DATE_TAG}_gsm8k_8b_g${GEN_LENGTH}_${label}_seed${SEED}"
  output_dir="outputs/${MODEL_NAME}/${run_name}"
  acc_file="$output_dir/accuracy.txt"
  if [ -f "$acc_file" ]; then
    vote=$(grep "Vote answer" "$acc_file" | grep -oE '[0-9]+\.[0-9]+%' || echo "N/A")
    final=$(grep "Final answer" "$acc_file" | grep -oE '[0-9]+\.[0-9]+%' || echo "N/A")
    printf "  %-22s vote=%-8s final=%-8s  (score=%s gamma=%s)\n" \
      "$label" "$vote" "$final" "$tscore" "$kl_gamma_val"
  else
    printf "  %-22s NO RESULT\n" "$label"
  fi
done
echo "============================================================"
