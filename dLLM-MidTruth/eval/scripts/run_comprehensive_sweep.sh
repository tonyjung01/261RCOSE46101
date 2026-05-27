#!/usr/bin/env bash
# Comprehensive ordering-formula sweep: 12 items, 113 runs total.
# Items run sequentially; each run uses all 4 GPUs via torchrun.
# Skip logic skips already-completed results (accuracy.txt exists).
#
# Items:
#  ①  margexkl γ=0.3   × 8B  g128,         4 tasks        (  4 runs)
#  ②  top1_prob         × 8B  g256/512,      4 tasks        (  8 runs)
#  ③  top1_prob         × 1.5 g128/256/512,  4 tasks        ( 12 runs)
#  ④  prob_margin       × 1.5 g128/256/512,  svamp/math/cd  (  9 runs)
#  ⑤  margexkl γ=0.5   × 8B  g256/512,      4 tasks        (  8 runs)
#  ⑥  margexkl γ=0.5   × 1.5 g128/256/512,  4 tasks        ( 12 runs)
#  ⑦  margexkl γ=0.3   × 8B  g256/512,      4 tasks        (  8 runs)
#  ⑧  margexkl γ=0.3   × 1.5 g128/256/512,  4 tasks        ( 12 runs)
#  ⑨  top1_heavy_blend  × 8B  g256/512,      4 tasks        (  8 runs)
#  ⑩  top1_heavy_blend  × 1.5 g128/256/512,  4 tasks        ( 12 runs)
#  ⑪  top1_x_margin     × 8B  g256/512,      4 tasks        (  8 runs)
#  ⑫  top1_x_margin     × 1.5 g128/256/512,  4 tasks        ( 12 runs)
#                                                      total: 113 runs
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
MASTER_PORT_BASE="${MASTER_PORT_BASE:-30000}"

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

port="$MASTER_PORT_BASE"
declare -a ALL_RUNS

# run_one <task> <label> <transfer> <model_name> <model_path> <model_tag> <gen_length>
#         [kl_gamma=1.0] [t_lambda=0.0] [t_tau=0.0]
run_one() {
  local task="$1"
  local label="$2"
  local transfer="$3"
  local model_name="$4"
  local model_path="$5"
  local model_tag="$6"
  local gen_length="$7"
  local kl_gamma="${8:-1.0}"
  local t_lambda="${9:-0.0}"
  local t_tau="${10:-0.0}"

  local alias
  alias=$(task_alias "$task")
  local run_name="${DATE_TAG}_${alias}_${model_tag}_g${gen_length}_${label}_seed${SEED}"
  local output_dir="outputs/${model_name}/${run_name}"
  local log_file="logs/${run_name}.log"
  local subset_file
  subset_file=$(subset_for_task "$task")

  ALL_RUNS+=("${model_name}|${run_name}")

  if [ -f "$output_dir/accuracy.txt" ]; then
    echo "[skip] $run_name"
    port=$((port + 1))
    return 0
  fi

  echo "============================================================"
  echo "[run ] $run_name"
  echo "       task=$task  model=$model_name  gen=$gen_length  transfer=$transfer"
  case "$transfer" in
    margin_exp_kl)         echo "       kl_gamma=$kl_gamma" ;;
    gated_temporal_margin) echo "       lambda=$t_lambda  tau=$t_tau" ;;
  esac
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
     TRANSFER_SCORE="$transfer" \
     KL_GAMMA="$kl_gamma" \
     TEMPORAL_LAMBDA="$t_lambda" \
     TEMPORAL_TAU="$t_tau" \
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

# ─────────────────────────────────────────────────────────────────────────────
sec "① margexkl γ=0.3 × LLaDA-8B g128, 4 tasks (4 runs)"
for task in gsm8k svamp math countdown; do
  run_one "$task" margexkl_g03 margin_exp_kl \
    "$MODEL_8B_NAME" "$MODEL_8B_PATH" 8b 128 0.3
done

# ─────────────────────────────────────────────────────────────────────────────
sec "② top1_prob × LLaDA-8B g256/512, 4 tasks (8 runs)"
for gl in 256 512; do
  for task in gsm8k svamp math countdown; do
    run_one "$task" top1prob top1_prob \
      "$MODEL_8B_NAME" "$MODEL_8B_PATH" 8b "$gl"
  done
done

# ─────────────────────────────────────────────────────────────────────────────
sec "③ top1_prob × LLaDA-1.5 g128/256/512, 4 tasks (12 runs)"
for gl in 128 256 512; do
  for task in gsm8k svamp math countdown; do
    run_one "$task" top1prob top1_prob \
      "$MODEL_15_NAME" "$MODEL_15_PATH" 15 "$gl"
  done
done

# ─────────────────────────────────────────────────────────────────────────────
sec "④ prob_margin × LLaDA-1.5 g128/256/512, svamp/math/countdown (9 runs)"
for gl in 128 256 512; do
  for task in svamp math countdown; do
    run_one "$task" probmargin prob_margin \
      "$MODEL_15_NAME" "$MODEL_15_PATH" 15 "$gl"
  done
done

# ─────────────────────────────────────────────────────────────────────────────
sec "⑤ margexkl γ=0.5 × LLaDA-8B g256/512, 4 tasks (8 runs)"
for gl in 256 512; do
  for task in gsm8k svamp math countdown; do
    run_one "$task" margexkl_g05 margin_exp_kl \
      "$MODEL_8B_NAME" "$MODEL_8B_PATH" 8b "$gl" 0.5
  done
done

# ─────────────────────────────────────────────────────────────────────────────
sec "⑥ margexkl γ=0.5 × LLaDA-1.5 g128/256/512, 4 tasks (12 runs)"
for gl in 128 256 512; do
  for task in gsm8k svamp math countdown; do
    run_one "$task" margexkl_g05 margin_exp_kl \
      "$MODEL_15_NAME" "$MODEL_15_PATH" 15 "$gl" 0.5
  done
done

# ─────────────────────────────────────────────────────────────────────────────
sec "⑦ margexkl γ=0.3 × LLaDA-8B g256/512, 4 tasks (8 runs)"
for gl in 256 512; do
  for task in gsm8k svamp math countdown; do
    run_one "$task" margexkl_g03 margin_exp_kl \
      "$MODEL_8B_NAME" "$MODEL_8B_PATH" 8b "$gl" 0.3
  done
done

# ─────────────────────────────────────────────────────────────────────────────
sec "⑧ margexkl γ=0.3 × LLaDA-1.5 g128/256/512, 4 tasks (12 runs)"
for gl in 128 256 512; do
  for task in gsm8k svamp math countdown; do
    run_one "$task" margexkl_g03 margin_exp_kl \
      "$MODEL_15_NAME" "$MODEL_15_PATH" 15 "$gl" 0.3
  done
done

# ─────────────────────────────────────────────────────────────────────────────
sec "⑨ top1_heavy_blend × LLaDA-8B g256/512, 4 tasks (8 runs)"
for gl in 256 512; do
  for task in gsm8k svamp math countdown; do
    run_one "$task" top1heavyblend top1_heavy_blend \
      "$MODEL_8B_NAME" "$MODEL_8B_PATH" 8b "$gl"
  done
done

# ─────────────────────────────────────────────────────────────────────────────
sec "⑩ top1_heavy_blend × LLaDA-1.5 g128/256/512, 4 tasks (12 runs)"
for gl in 128 256 512; do
  for task in gsm8k svamp math countdown; do
    run_one "$task" top1heavyblend top1_heavy_blend \
      "$MODEL_15_NAME" "$MODEL_15_PATH" 15 "$gl"
  done
done

# ─────────────────────────────────────────────────────────────────────────────
sec "⑪ top1_x_margin × LLaDA-8B g256/512, 4 tasks (8 runs)"
for gl in 256 512; do
  for task in gsm8k svamp math countdown; do
    run_one "$task" top1xmargin top1_x_margin \
      "$MODEL_8B_NAME" "$MODEL_8B_PATH" 8b "$gl"
  done
done

# ─────────────────────────────────────────────────────────────────────────────
sec "⑫ top1_x_margin × LLaDA-1.5 g128/256/512, 4 tasks (12 runs)"
for gl in 128 256 512; do
  for task in gsm8k svamp math countdown; do
    run_one "$task" top1xmargin top1_x_margin \
      "$MODEL_15_NAME" "$MODEL_15_PATH" 15 "$gl"
  done
done

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "############################################################"
echo "# COMPREHENSIVE SWEEP SUMMARY  (DATE=${DATE_TAG})"
echo "############################################################"
prev_model=""
for entry in "${ALL_RUNS[@]}"; do
  IFS='|' read -r model_name run_name <<< "$entry"
  if [ "$model_name" != "$prev_model" ]; then
    echo "  --- $model_name ---"
    prev_model="$model_name"
  fi
  output_dir="outputs/${model_name}/${run_name}"
  if [ -f "$output_dir/accuracy.txt" ]; then
    vote=$(grep "Vote answer"   "$output_dir/accuracy.txt" | grep -oE '[0-9]+\.[0-9]+%' || echo "N/A")
    final=$(grep "Final answer" "$output_dir/accuracy.txt" | grep -oE '[0-9]+\.[0-9]+%' || echo "N/A")
    printf "  %-60s vote=%-9s final=%s\n" "$run_name" "$vote" "$final"
  else
    printf "  %-60s NO RESULT\n" "$run_name"
  fi
done
echo "############################################################"
echo "# Done. Total runs tracked: ${#ALL_RUNS[@]}"
echo "############################################################"
