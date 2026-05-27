#!/usr/bin/env bash
# =============================================================================
# Token-ordering experiment - full pipeline for ONE generation length.
#
# Reproduces "앞선 코드" for a single seq length:
#   - run_gsm8k_gated_sweep.sh        (GSM8K: prob_margin baseline + 5 configs)
#   - run_gsm8k_gated_sweep_extra.sh  (GSM8K: 5 more configs)
#   - launch_candidate_all_tasks_balanced_tmux.sh
#                                     (l005_t012 candidate on svamp/math/countdown)
#
# Every eval run uses ALL 4 GPUs (data-parallel via torchrun --nproc_per_node 4).
# Runs are executed sequentially inside this script; combined with tsp -S 1
# only one 4-GPU job is active at a time.
#
# Required env : GEN_LENGTH   TAG
# Optional env : MASTER_PORT_BASE   PYTHON_BIN
# =============================================================================
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EVAL_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$EVAL_DIR"

: "${GEN_LENGTH:?set GEN_LENGTH (e.g. 128 / 256 / 512)}"
: "${TAG:?set TAG (e.g. g128 / g256 / g512)}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29500}"
export PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/anaconda3/envs/tiaf/bin/python}"

GPUS=(0 1 2 3)            # all 4 GPUs per run
SEED=42
BATCH_SIZE=4
BLOCK_LENGTH=32
TOKEN_PER_STEP=2          # diffusion_steps = GEN_LENGTH / TOKEN_PER_STEP
VOTE_METHOD=exp
ALPHA=5.0
SAVE_VOTE_DEBUG=true

mkdir -p analysis/transfer_score_probe logs outputs/LLaDA-1.5

# ---- subset index files (full dataset) --------------------------------------
GSM_SUBSET="analysis/transfer_score_probe/full_gsm8k_indices.txt"
SVAMP_SUBSET="analysis/transfer_score_probe/full_svamp_indices.txt"
MATH_SUBSET="analysis/transfer_score_probe/full_math500_indices.txt"
CD_SUBSET="analysis/transfer_score_probe/full_countdown_indices.txt"
printf "%s\n" $(seq 0 1318) > "$GSM_SUBSET"
printf "%s\n" $(seq 0 299)  > "$SVAMP_SUBSET"
printf "%s\n" $(seq 0 499)  > "$MATH_SUBSET"
printf "%s\n" $(seq 0 255)  > "$CD_SUBSET"

if [ ! -f ../dataset/countdown_cd3_test.jsonl ]; then
  echo "[fatal] missing dataset/countdown_cd3_test.jsonl"
  exit 1
fi

# ---- 10 GSM8K gated_temporal_margin configs (sweep + extra-sweep merged) -----
# label:lambda:tau
GSM8K_CONFIGS=(
  "l005_t015:0.05:0.15"
  "l010_t012:0.10:0.12"
  "l008_t012:0.08:0.12"
  "l005_t012:0.05:0.12"
  "l003_t010:0.03:0.10"
  "l008_t010:0.08:0.10"
  "l005_t010:0.05:0.10"
  "l003_t012:0.03:0.12"
  "l008_t008:0.08:0.08"
  "l005_t008:0.05:0.08"
)

idx=0
run_eval() {
  # $1 task  $2 subset_file  $3 run_name  $4 transfer_score  $5 lambda  $6 tau
  local task="$1" subset="$2" run_name="$3" transfer="$4" lam="$5" tau="$6"
  local out_dir="outputs/LLaDA-1.5/${run_name}"
  local log_file="logs/${run_name}.log"
  local port=$((MASTER_PORT_BASE + idx))
  idx=$((idx + 1))

  echo "============================================================"
  echo "[${TAG}] ${run_name}"
  echo "  task=${task} transfer=${transfer} lambda=${lam} tau=${tau}"
  echo "  gen_length=${GEN_LENGTH} gpus=${GPUS[*]} port=${port}"
  echo "============================================================"

  if [ -f "${out_dir}/accuracy.txt" ]; then
    echo "[skip] result already exists: ${out_dir}/accuracy.txt"
    return 0
  fi

  if env \
      PYTHONFAULTHANDLER=1 \
      PYTHONUNBUFFERED=1 \
      NCCL_ASYNC_ERROR_HANDLING=1 \
      MASTER_PORT="${port}" \
      PYTHON_BIN="${PYTHON_BIN}" \
      SUBSET_INDICES_FILE="${subset}" \
      RUN_NAME="${run_name}" \
      TASK="${task}" \
      GEN_LENGTH="${GEN_LENGTH}" \
      BLOCK_LENGTH="${BLOCK_LENGTH}" \
      TOKEN_PER_STEP="${TOKEN_PER_STEP}" \
      VOTE_METHOD="${VOTE_METHOD}" \
      ALPHA="${ALPHA}" \
      TRANSFER_SCORE="${transfer}" \
      TEMPORAL_LAMBDA="${lam}" \
      TEMPORAL_TAU="${tau}" \
      SEED="${SEED}" \
      BATCH_SIZE="${BATCH_SIZE}" \
      SAVE_VOTE_DEBUG="${SAVE_VOTE_DEBUG}" \
      bash scripts/run_retry_policy_experiment.sh "${GPUS[@]}" \
      2>&1 | tee "${log_file}"; then
    echo "[ok] ${run_name}"
  else
    echo "[ERROR] run failed (continuing with next): ${run_name}"
  fi
}

echo "############################################################"
echo "# Experiment ${TAG} : gen_length=${GEN_LENGTH}, 4-GPU per run"
echo "# diffusion_steps=$((GEN_LENGTH / TOKEN_PER_STEP))  block_length=${BLOCK_LENGTH}"
echo "############################################################"

# 1) GSM8K prob_margin baseline
run_eval gsm8k "$GSM_SUBSET" "${TAG}_gsm8k_full_exp_bs4_probmargin" prob_margin 0.1 0.15

# 2) GSM8K gated_temporal_margin sweep (10 configs)
for spec in "${GSM8K_CONFIGS[@]}"; do
  label="${spec%%:*}"; rest="${spec#*:}"; lam="${rest%%:*}"; tau="${rest##*:}"
  run_eval gsm8k "$GSM_SUBSET" "${TAG}_gsm8k_full_exp_bs4_gatedtm_${label}" \
    gated_temporal_margin "$lam" "$tau"
done

# 3) best candidate l005_t012 on the other three tasks
run_eval svamp     "$SVAMP_SUBSET" "${TAG}_svamp_full_exp_bs4_gatedtm_l005_t012"     gated_temporal_margin 0.05 0.12
run_eval math      "$MATH_SUBSET"  "${TAG}_math500_full_exp_bs4_gatedtm_l005_t012"   gated_temporal_margin 0.05 0.12
run_eval countdown "$CD_SUBSET"    "${TAG}_countdown_full_exp_bs4_gatedtm_l005_t012" gated_temporal_margin 0.05 0.12

# ---- summary ----------------------------------------------------------------
echo ""
echo "############################################################"
echo "# SUMMARY ${TAG} (gen_length=${GEN_LENGTH})"
echo "############################################################"
for d in outputs/LLaDA-1.5/${TAG}_*; do
  [ -f "$d/accuracy.txt" ] || continue
  vote=$(grep -m1 "Vote answer" "$d/accuracy.txt" | grep -oE '[0-9]+\.[0-9]+%' || true)
  final=$(grep -m1 "Final answer" "$d/accuracy.txt" | grep -oE '[0-9]+\.[0-9]+%' || true)
  printf "  %-54s vote=%-9s final=%-9s\n" "$(basename "$d")" "${vote:-NA}" "${final:-NA}"
done
echo "############################################################"
echo "# Experiment ${TAG} finished."
echo "############################################################"
