#!/bin/bash

model_name=TGAT
decoder_name=mlp
dataset=dsub1m
date=1225
gt_weight=0.5

# 支持多个 alpha 搜索
alphas=(0.1 0.5 0.9)

method=ptcl
gpus=(0 1 2 3)
max_tasks_per_gpu=1    # 每 GPU 同时任务数上限

# =======================
# 构建任务队列：(alpha, seed)
# =======================
declare -a task_queue
for alpha in "${alphas[@]}"; do
  for seed in {0..4}; do
    task_queue+=("$alpha $seed")
  done
done

# 每 GPU 的任务统计 + PID 列表
declare -A gpu_task_count
declare -A gpu_task_pids

for gpu in "${gpus[@]}"; do
  gpu_task_count[$gpu]=0
  gpu_task_pids[$gpu]=""
done

# =======================
# 运行单个任务
# =======================
assign_task_to_gpu() {
  local gpu=$1
  local alpha=$2
  local seed=$3
  local next_seed=$((seed + 1))

  local prefix="${date}_${method}_${model_name}_${decoder_name}_${dataset}_${gt_weight}_${alpha}"
  local result_file="results/${prefix}_results.txt"

  mkdir -p results
  if [[ ! -f $result_file ]]; then
    echo "AUC   ACC" > $result_file
  fi

  echo "启动任务: GPU=$gpu, alpha=$alpha, seed=$seed"

  python train.py \
        --prefix $prefix \
        --method $method \
        --start_runs $seed \
        --end_runs $next_seed \
        --gpu $gpu \
        --mmodel_name $model_name \
        --emodel_name $decoder_name \
        --use_ps_back 1 \
        --alpha $alpha \
        --decoder 1 \
        --num_em_iters 30 \
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

# =======================
# 主循环：动态调度 GPU
# =======================
while [[ ${#task_queue[@]} -gt 0 ]]; do
  for gpu in "${gpus[@]}"; do
    if [[ ${gpu_task_count[$gpu]} -lt $max_tasks_per_gpu ]]; then
      
      if [[ ${#task_queue[@]} -gt 0 ]]; then
        task="${task_queue[0]}"
        task_queue=("${task_queue[@]:1}")

        read alpha seed <<< "$task"
        assign_task_to_gpu $gpu $alpha $seed
      fi
    fi
  done

  sleep 1

  # 检查 PID 释放 GPU
  for gpu in "${gpus[@]}"; do
    for pid in ${gpu_task_pids[$gpu]}; do
      if ! ps -p $pid > /dev/null; then
        gpu_task_count[$gpu]=$((gpu_task_count[$gpu] - 1))
        gpu_task_pids[$gpu]=${gpu_task_pids[$gpu]//$pid/}
      fi
    done
  done
done

# 等待所有任务收尾
for gpu in "${gpus[@]}"; do
  for pid in ${gpu_task_pids[$gpu]}; do
    if [[ -n $pid ]]; then
      wait $pid
    fi
  done
done

echo "所有 alpha × seed 任务已完成。"
