#!/usr/bin/env bash
# exp_kl_decay sweep: score = exp(-gamma * KL(p_curr || p_prev))
# (margin_exp_kl without the margin factor)
#
# Grid: gamma in {0.5, 1.0, 2.0} x model in {8B, 1.5} x gen in {128, 256, 512}
#       x task in {gsm8k, svamp, math, countdown}
#       = 3 * 2 * 3 * 4 = 72 runs
#
# Loop order (outer -> inner): gen length asc, model, task, gamma.
# So all g128 runs finish before any g256, then g512.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EVAL_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$EVAL_DIR"

if [ "$#" -gt 0 ]; then
  GPU_ARGS=("$@")
else
  GPU_ARGS=(0 1 2 3)
fi

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/anaconda3/envs/tiaf/bin/python}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-4}"
VOTE_METHOD="${VOTE_METHOD:-exp}"
ALPHA="${ALPHA:-5.0}"
BLOCK_LENGTH="${BLOCK_LENGTH:-32}"
TOKEN_PER_STEP="${TOKEN_PER_STEP:-2}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-30400}"

MODEL_8B_PATH="/home/ubuntu/pbm/261RCOSE46101/dLLM-MidTruth/pretrained_weights/GSAI-ML/LLaDA-8B-Instruct"
MODEL_8B_NAME="LLaDA-8B-Instruct"
MODEL_15_PATH="/home/ubuntu/pbm/261RCOSE46101/dLLM-MidTruth/pretrained_weights/GSAI-ML/LLaDA-1.5"
MODEL_15_NAME="LLaDA-1.5"

mkdir -p analysis/transfer_score_probe logs \
         "outputs/$MODEL_8B_NAME" "outputs/$MODEL_15_NAME"

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

gamma_label() {
  case "$1" in
    0.5) echo "g05" ;;
    1.0) echo "g1"  ;;
    2.0) echo "g2"  ;;
    0.3) echo "g03" ;;
    *) echo "g$(echo "$1" | tr '.' '_')" ;;
  esac
}

port="$MASTER_PORT_BASE"

# run_one <task> <model_name> <model_path> <model_tag> <gen_length> <kl_gamma>
run_one() {
  local task="$1"
  local model_name="$2"
  local model_path="$3"
  local model_tag="$4"
  local gen_length="$5"
  local kl_gamma="$6"

  local glabel
  glabel=$(gamma_label "$kl_gamma")
  local label="expkldecay_${glabel}"
  local alias
  alias=$(task_alias "$task")
  local run_name="${DATE_TAG}_${alias}_${model_tag}_g${gen_length}_${label}_seed${SEED}"
  local output_dir="outputs/${model_name}/${run_name}"
  local log_file="logs/${run_name}.log"
  local subset_file
  subset_file=$(subset_for_task "$task")

  if [ -f "$output_dir/accuracy.txt" ]; then
    echo "[skip] $run_name"
    port=$((port + 1))
    return 0
  fi

  echo "============================================================"
  echo "[run ] $run_name"
  echo "       task=$task  model=$model_name  gen=$gen_length"
  echo "       transfer=exp_kl_decay  kl_gamma=$kl_gamma"
  echo "       port=$port  gpus=${GPU_ARGS[*]}"
  echo "============================================================"

  if PYTHONFAULTHANDLER=1 \
     PYTHONUNBUFFERED=1 \
     MASTER_PORT="$port" \
     PYTHON_BIN="$PYTHON_BIN" \
     MODEL_PATH="$model_path" \
     MODEL_NAME="$model_name" \
     SUBSET_INDICES_FILE="$subset_file" \
     RUN_NAME="$run_name" \
     TASK="$task" \
     VOTE_METHOD="$VOTE_METHOD" \
     ALPHA="$ALPHA" \
     GEN_LENGTH="$gen_length" \
     BLOCK_LENGTH="$BLOCK_LENGTH" \
     TOKEN_PER_STEP="$TOKEN_PER_STEP" \
     TRANSFER_SCORE="exp_kl_decay" \
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

GAMMAS=(0.5 1.0 2.0)
TASKS=(gsm8k svamp math countdown)
MODELS=(
  "$MODEL_8B_NAME|$MODEL_8B_PATH|8b"
  "$MODEL_15_NAME|$MODEL_15_PATH|15"
)

for gl in 128 256 512; do
  sec "exp_kl_decay sweep: gen_length=$gl  (2 models x 4 tasks x 3 gammas = 24 runs)"
  for model_spec in "${MODELS[@]}"; do
    IFS='|' read -r mname mpath mtag <<< "$model_spec"
    for task in "${TASKS[@]}"; do
      for gamma in "${GAMMAS[@]}"; do
        run_one "$task" "$mname" "$mpath" "$mtag" "$gl" "$gamma"
      done
    done
  done
done

echo ""
echo "exp_kl_decay sweep done."
