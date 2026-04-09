#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com

# Configuration variables
GPU_IDS=(0 1 2 3)  

MASTER_PORT=29413

# Arrays of tasks and generation lengths
TASKS=("gsm8k" "math" "countdown" "svamp")
GEN_LENGTHS=(128 256 512)
token_per_step=2
temperature=0.0

# Model to be used for evaluation
model_path="/mnt/nas/share/weights/LLaDA-8B-Instruct"

# Vote Settings
enable_vote=true
vote_method="exp"
alpha=5

# Set GPU IDs from command line if provided
if [ $# -gt 0 ]; then
  # Clear default GPU list and add provided GPUs
  GPU_IDS=()
  for arg in "$@"; do
    GPU_IDS+=("$arg")
  done
fi

GPU_LIST=$(IFS=,; echo "${GPU_IDS[*]}")
NUM_GPUS=${#GPU_IDS[@]}
echo "Using GPUs: $GPU_LIST (nproc_per_node=$NUM_GPUS)"

for task in "${TASKS[@]}"; do
  for gen_length in "${GEN_LENGTHS[@]}"; do
    # Set batch size based on generation length
    if [ "$gen_length" == "256" ] || [ "$gen_length" == "512" ]; then
      batch_size=4
    else
      batch_size=8
    fi


    if [ "$task" == "math" ]; then
      batch_size=$((batch_size / 4))
    fi


    diffusion_steps=$((gen_length / token_per_step))
    
    echo "Evaluating task: $task with generation length: $gen_length and diffusion steps: $diffusion_steps (batch size: $batch_size)"
    
    if [ "$enable_vote" = true ]; then
      CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun \
        --nproc_per_node $NUM_GPUS \
        --master_port $MASTER_PORT \
        eval.py \
        --dataset $task \
        --batch_size $batch_size \
        --gen_length $gen_length \
        --diffusion_steps $diffusion_steps \
        --temperature $temperature \
        --output_dir "outputs/${model}/${task}_gen_${gen_length}_steps_${diffusion_steps}_temp_${temperature}_vote" \
        --model_path $model_path \
        --model_name $model \
        --enable_vote \
        --vote_method $vote_method \
        --alpha $alpha
    else
      CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun \
        --nproc_per_node $NUM_GPUS \
        --master_port $MASTER_PORT \
        eval.py \
        --dataset $task \
        --batch_size $batch_size \
        --gen_length $gen_length \
        --diffusion_steps $diffusion_steps \
        --temperature $temperature \
        --output_dir "outputs/${model}/${task}_gen_${gen_length}_steps_${diffusion_steps}_temp_${temperature}" \
        --model_path $model_path \
        --model_name $model
    fi
  done
done


echo "All evaluations completed!"
