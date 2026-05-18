#!/bin/bash

model_name=TGAT
dataset=arxiv
date=1126
# graphmixer 0.2 0.4 0.8
# tgat 0.2 0.3 0.4 0.7 0.8
# tcl 
# tgn 0.1 0.2 0.3 0.4 0.6 0.8
# dygformer 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
gt_weight=0.7
alphas=(0.2)
gpus=(3)  # List of GPUs
max_tasks_per_gpu=5  # Maximum number of tasks per GPU

# Define task queue
declare -a task_queue
for alpha in "${alphas[@]}"; do
  for seed in {0..4}; do
    task_queue+=("$alpha $seed")
  done
done

# Track tasks for each GPU
declare -A gpu_task_count
declare -A gpu_task_pids

for gpu in "${gpus[@]}"; do
  gpu_task_count[$gpu]=0
  gpu_task_pids[$gpu]=""
done

# Assign task to GPU
assign_task_to_gpu() {
  local gpu=$1
  local alpha=$2
  local seed=$3
  local next_seed=$((seed + 1))
  local prefix="${date}_pb_1d_${model_name}_${dataset}_gt_${gt_weight}_alpha_${alpha}"
  local result_file="results/${prefix}_results.txt"

  mkdir -p results
  if [[ ! -f $result_file ]]; then
    echo "AUC   ACC  F1  AP" > $result_file
  fi

  echo "Assign Task: GPU $gpu, alpha=$alpha, seed=$seed"

  # Run the Python script
  python train_ncem.py \
    --prefix $prefix \
    --method $method \
    --start_runs $seed \
    --end_runs $next_seed \
    --gpu $gpu \
    --mmodel_name $model_name \
    --emodel_name mlp \
    --use_ps_back 1 \
    --decoder 1 \
    --num_em_iters 30\
    --num_epochs_m_step 50 \
    --num_epochs_e_step 100 \
    --num_epochs_e_warmup 100 \
    --num_epochs_m_warmup 100 \
    --iter_patience 5 \
    --patience 15 \
    --warmup_e_train 0 \
    --warmup_m_train 0 \
    --gt_weight $gt_weight \
    --dataset_name $dataset 2>/dev/null | tail -n 1 >> $result_file &

  local pid=$!
  gpu_task_count[$gpu]=$((gpu_task_count[$gpu] + 1))
  gpu_task_pids[$gpu]+=" $pid"
}

# Distribute tasks
while [[ ${#task_queue[@]} -gt 0 ]]; do
  for gpu in "${gpus[@]}"; do
    # Check if the current GPU can accept more tasks
    if [[ ${gpu_task_count[$gpu]} -lt $max_tasks_per_gpu ]]; then
      if [[ ${#task_queue[@]} -gt 0 ]]; then
        # Dequeue the first task
        task="${task_queue[0]}"
        task_queue=("${task_queue[@]:1}")

        # Parse task parameters and assign to GPU
        read alpha seed <<< "$task"
        assign_task_to_gpu $gpu $alpha $seed
      fi
    fi
  done
  sleep 1  # Check every second

  # Check for completed tasks and update GPU task count
  for gpu in "${gpus[@]}"; do
    for pid in ${gpu_task_pids[$gpu]}; do
      if ! ps -p $pid > /dev/null; then
        gpu_task_count[$gpu]=$((gpu_task_count[$gpu] - 1))
        gpu_task_pids[$gpu]=${gpu_task_pids[$gpu]//$pid/}
      fi
    done
  done
done

# Wait for all background tasks to complete
for gpu in "${gpus[@]}"; do
  for pid in ${gpu_task_pids[$gpu]}; do
    if [[ -n $pid ]]; then
      wait $pid
    fi
  done
done
