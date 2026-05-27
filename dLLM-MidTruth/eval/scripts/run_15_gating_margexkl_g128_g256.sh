#!/usr/bin/env bash
# LLaDA-1.5 sweep: gated (l=0.05,t=0.15), gated (l=0.10,t=0.15), margexkl gamma=0.3
# x gen in {128, 256} x 4 tasks = 24 runs.
# DATE_TAG defaults to 20260524 so already-done runs (e.g., gsm8k g128 margexkl_g03)
# are picked up by the skip logic.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EVAL_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$EVAL_DIR"

if [ "$#" -gt 0 ]; then
  GPU_ARGS=("$@")
else
  GPU_ARGS=(0 1 2 3)
fi

DATE_TAG="${DATE_TAG:-20260524}"
PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/anaconda3/envs/tiaf/bin/python}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-4}"
VOTE_METHOD="${VOTE_METHOD:-exp}"
ALPHA="${ALPHA:-5.0}"
BLOCK_LENGTH="${BLOCK_LENGTH:-32}"
TOKEN_PER_STEP="${TOKEN_PER_STEP:-2}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-30600}"

MODEL_15_PATH="/home/ubuntu/pbm/261RCOSE46101/dLLM-MidTruth/pretrained_weights/GSAI-ML/LLaDA-1.5"
MODEL_15_NAME="LLaDA-1.5"

mkdir -p analysis/transfer_score_probe logs "outputs/$MODEL_15_NAME"

GSM_SUBSET="$EVAL_DIR/analysis/transfer_score_probe/full_gsm8k_indices.txt"
SVAMP_SUBSET="$EVAL_DIR/analysis/transfer_score_probe/full_svamp_indices.txt"
MATH_SUBSET="$EVAL_DIR/analysis/transfer_score_probe/full_math500_indices.txt"
CD_SUBSET="$EVAL_DIR/analysis/transfer_score_probe/full_countdown_indices.txt"

printf "%s\n" $(seq 0 1318) > "$GSM_SUBSET"
printf "%s\n" $(seq 0 299)  > "$SVAMP_SUBSET"
printf "%s\n" $(seq 0 499)  > "$MATH_SUBSET"
printf "%s\n" $(seq 0 255)  > "$CD_SUBSET"

subset_for_task() {
  case "$1" in
    gsm8k)     echo "$GSM_SUBSET" ;;
    svamp)     echo "$SVAMP_SUBSET" ;;
    math)      echo "$MATH_SUBSET" ;;
    countdown) echo "$CD_SUBSET" ;;
    *) echo "unknown task: $1" >&2; return 1 ;;
  esac
}

task_alias() {
  case "$1" in
    math) echo "math500" ;;
    *) echo "$1" ;;
  esac
}

port="$MASTER_PORT_BASE"

# run_one <task> <label> <transfer> <gen_length> [extra1] [extra2]
#   transfer=gated_temporal_margin -> extra1=lambda, extra2=tau
#   transfer=margin_exp_kl         -> extra1=kl_gamma
run_one() {
  local task="$1"
  local label="$2"
  local transfer="$3"
  local gen_length="$4"
  local e1="${5:-0.0}"
  local e2="${6:-0.0}"

  local alias
  alias=$(task_alias "$task")
  local run_name="${DATE_TAG}_${alias}_15_g${gen_length}_${label}_seed${SEED}"
  local output_dir="outputs/${MODEL_15_NAME}/${run_name}"
  local log_file="logs/${run_name}.log"
  local subset_file
  subset_file=$(subset_for_task "$task")

  if [ -f "$output_dir/accuracy.txt" ]; then
    echo "[skip] $run_name"
    port=$((port + 1))
    return 0
  fi

  local t_lambda="0.0" t_tau="0.0" kl_gamma="1.0"
  case "$transfer" in
    gated_temporal_margin) t_lambda="$e1"; t_tau="$e2" ;;
    margin_exp_kl)         kl_gamma="$e1" ;;
  esac

  echo "============================================================"
  echo "[run ] $run_name"
  echo "       task=$task  gen=$gen_length  transfer=$transfer"
  case "$transfer" in
    gated_temporal_margin) echo "       lambda=$t_lambda  tau=$t_tau" ;;
    margin_exp_kl)         echo "       kl_gamma=$kl_gamma" ;;
  esac
  echo "       port=$port  gpus=${GPU_ARGS[*]}"
  echo "============================================================"

  if PYTHONFAULTHANDLER=1 \
     PYTHONUNBUFFERED=1 \
     MASTER_PORT="$port" \
     PYTHON_BIN="$PYTHON_BIN" \
     MODEL_PATH="$MODEL_15_PATH" \
     MODEL_NAME="$MODEL_15_NAME" \
     SUBSET_INDICES_FILE="$subset_file" \
     RUN_NAME="$run_name" \
     TASK="$task" \
     VOTE_METHOD="$VOTE_METHOD" \
     ALPHA="$ALPHA" \
     GEN_LENGTH="$gen_length" \
     BLOCK_LENGTH="$BLOCK_LENGTH" \
     TOKEN_PER_STEP="$TOKEN_PER_STEP" \
     TRANSFER_SCORE="$transfer" \
     TEMPORAL_LAMBDA="$t_lambda" \
     TEMPORAL_TAU="$t_tau" \
     KL_GAMMA="$kl_gamma" \
     SEED="$SEED" \
     BATCH_SIZE="$BATCH_SIZE" \
     SAVE_VOTE_DEBUG="true" \
       bash scripts/run_retry_policy_experiment.sh "${GPU_ARGS[@]}" \
       2>&1 | tee "$log_file"; then
    echo "[ok ] $run_name"
  else
    echo "[ERR] run failed (continuing): $run_name"
  fi

  port=$((port + 1))
}

sec() {
  echo ""
  echo "############################################################"
  printf "# %s\n" "$*"
  echo "############################################################"
}

for gl in 128 256; do
  sec "1.5 gating + margexkl_g03 sweep: gen_length=$gl (3 orderings x 4 tasks = 12 runs)"
  for task in gsm8k svamp math countdown; do
    run_one "$task" gated_l005_t015 gated_temporal_margin "$gl" 0.05 0.15
    run_one "$task" gated_l010_t015 gated_temporal_margin "$gl" 0.10 0.15
    run_one "$task" margexkl_g03    margin_exp_kl         "$gl" 0.3
  done
done

echo ""
echo "1.5 gating + margexkl_g03 (g128/g256) sweep done."
