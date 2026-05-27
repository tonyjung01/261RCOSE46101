#!/usr/bin/env bash
# Ordering-formula sweep across all 4 tasks with LLaDA-8B-Instruct.
# 31 new runs: top1_prob/gated/margexkl_g0.5 on all tasks,
#              prob_margin/margexkl_g1/top1_heavy_blend/top1_x_margin/margin_heavy_blend
#              on SVAMP+MATH500+Countdown (GSM8K already done).
# Runs sequentially; skip logic skips already-completed results.
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
GEN_LENGTH="${GEN_LENGTH:-128}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29800}"

MODEL_PATH="${MODEL_PATH:-/home/ubuntu/pbm/261RCOSE46101/dLLM-MidTruth/pretrained_weights/GSAI-ML/LLaDA-8B-Instruct}"
MODEL_NAME="${MODEL_NAME:-LLaDA-8B-Instruct}"

mkdir -p analysis/transfer_score_probe logs "outputs/$MODEL_NAME"

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

# eval.py --dataset uses "math"; run names use "math500"
task_alias() {
  case "$1" in
    math) echo "math500" ;;
    *) echo "$1" ;;
  esac
}

port="$MASTER_PORT_BASE"
declare -a ALL_RUNS

# run_one <task> <label> <transfer_score> [kl_gamma=1.0] [t_lambda=0.0] [t_tau=0.0]
run_one() {
  local task="$1"
  local label="$2"
  local transfer="$3"
  local kl_gamma="${4:-1.0}"
  local t_lambda="${5:-0.0}"
  local t_tau="${6:-0.0}"

  local alias
  alias=$(task_alias "$task")
  local run_name="${DATE_TAG}_${alias}_8b_g${GEN_LENGTH}_${label}_seed${SEED}"
  local output_dir="outputs/${MODEL_NAME}/${run_name}"
  local log_file="logs/${run_name}.log"
  local subset_file
  subset_file=$(subset_for_task "$task")

  ALL_RUNS+=("$run_name")

  if [ -f "$output_dir/accuracy.txt" ]; then
    echo "[sweep] skip (already done): $run_name"
    port=$((port + 1))
    return 0
  fi

  echo "============================================================"
  echo "[sweep] task=$task  label=$label  transfer=$transfer"
  case "$transfer" in
    margin_exp_kl)          echo "         kl_gamma=$kl_gamma" ;;
    gated_temporal_margin)  echo "         lambda=$t_lambda  tau=$t_tau" ;;
  esac
  echo "         run=$run_name  port=$port  gpus=${GPU_ARGS[*]}"
  echo "============================================================"

  if PYTHONFAULTHANDLER=1 \
     PYTHONUNBUFFERED=1 \
     MASTER_PORT="$port" \
     PYTHON_BIN="$PYTHON_BIN" \
     MODEL_PATH="$MODEL_PATH" \
     MODEL_NAME="$MODEL_NAME" \
     SUBSET_INDICES_FILE="$subset_file" \
     RUN_NAME="$run_name" \
     TASK="$task" \
     VOTE_METHOD="$VOTE_METHOD" \
     ALPHA="$ALPHA" \
     GEN_LENGTH="$GEN_LENGTH" \
     TRANSFER_SCORE="$transfer" \
     KL_GAMMA="$kl_gamma" \
     TEMPORAL_LAMBDA="$t_lambda" \
     TEMPORAL_TAU="$t_tau" \
     SEED="$SEED" \
     BATCH_SIZE="$BATCH_SIZE" \
     SAVE_VOTE_DEBUG="true" \
       bash scripts/run_retry_policy_experiment.sh "${GPU_ARGS[@]}" \
       2>&1 | tee "$log_file"; then
    echo "[ok] $run_name"
  else
    echo "[ERROR] run failed (continuing): $run_name"
  fi

  port=$((port + 1))
}

echo "############################################################"
echo "# ORDERING SWEEP — all tasks, LLaDA-8B-Instruct, gen_length=${GEN_LENGTH}"
echo "# GPUs: ${GPU_ARGS[*]}  DATE: ${DATE_TAG}  total runs: 31"
echo "############################################################"

# top1_prob (4 runs — all tasks)
run_one gsm8k     top1prob  top1_prob
run_one svamp     top1prob  top1_prob
run_one math      top1prob  top1_prob
run_one countdown top1prob  top1_prob

# prob_margin (3 runs — GSM8K already done in run_ordering_sweep_8b.sh)
run_one svamp     probmargin  prob_margin
run_one math      probmargin  prob_margin
run_one countdown probmargin  prob_margin

# gated_temporal_margin λ=0.05 τ=0.15 (4 runs — all tasks)
run_one gsm8k     gated_l005_t015  gated_temporal_margin  1.0  0.05  0.15
run_one svamp     gated_l005_t015  gated_temporal_margin  1.0  0.05  0.15
run_one math      gated_l005_t015  gated_temporal_margin  1.0  0.05  0.15
run_one countdown gated_l005_t015  gated_temporal_margin  1.0  0.05  0.15

# gated_temporal_margin λ=0.10 τ=0.15 (4 runs — all tasks)
run_one gsm8k     gated_l010_t015  gated_temporal_margin  1.0  0.10  0.15
run_one svamp     gated_l010_t015  gated_temporal_margin  1.0  0.10  0.15
run_one math      gated_l010_t015  gated_temporal_margin  1.0  0.10  0.15
run_one countdown gated_l010_t015  gated_temporal_margin  1.0  0.10  0.15

# margin_exp_kl γ=0.5 (4 runs — all tasks)
run_one gsm8k     margexkl_g05  margin_exp_kl  0.5
run_one svamp     margexkl_g05  margin_exp_kl  0.5
run_one math      margexkl_g05  margin_exp_kl  0.5
run_one countdown margexkl_g05  margin_exp_kl  0.5

# margin_exp_kl γ=1 (3 runs — GSM8K already done)
run_one svamp     margexkl_g1  margin_exp_kl  1.0
run_one math      margexkl_g1  margin_exp_kl  1.0
run_one countdown margexkl_g1  margin_exp_kl  1.0

# top1_heavy_blend (3 runs — GSM8K already done)
run_one svamp     top1heavyblend  top1_heavy_blend
run_one math      top1heavyblend  top1_heavy_blend
run_one countdown top1heavyblend  top1_heavy_blend

# top1_x_margin (3 runs — GSM8K already done)
run_one svamp     top1xmargin  top1_x_margin
run_one math      top1xmargin  top1_x_margin
run_one countdown top1xmargin  top1_x_margin

# margin_heavy_blend (3 runs — GSM8K already done)
run_one svamp     marginheavyblend  margin_heavy_blend
run_one math      marginheavyblend  margin_heavy_blend
run_one countdown marginheavyblend  margin_heavy_blend

# Summary
echo ""
echo "############################################################"
echo "# SWEEP SUMMARY  (${DATE_TAG}, LLaDA-8B-Instruct, gen_length=${GEN_LENGTH})"
echo "############################################################"
for run_name in "${ALL_RUNS[@]}"; do
  output_dir="outputs/${MODEL_NAME}/${run_name}"
  if [ -f "$output_dir/accuracy.txt" ]; then
    vote=$(grep "Vote answer"  "$output_dir/accuracy.txt" | grep -oE '[0-9]+\.[0-9]+%' || echo "N/A")
    final=$(grep "Final answer" "$output_dir/accuracy.txt" | grep -oE '[0-9]+\.[0-9]+%' || echo "N/A")
    printf "  %-56s vote=%-9s final=%s\n" "$run_name" "$vote" "$final"
  else
    printf "  %-56s NO RESULT\n" "$run_name"
  fi
done
echo "############################################################"
