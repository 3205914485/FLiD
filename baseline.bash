#!/bin/bash
model=TGAT
dataset=dsub1m

# 指定 GPU 列表（可任意数量）
GPU_LIST=(0 1 2 3)

# 训练链接预测任务
seeds=(0 1 2 3 4)

for i in "${!seeds[@]}"; do
{
    seed=${seeds[$i]}
    next_seed=$((seed + 1))

    # 从 GPU 列表循环选择 GPU
    gpu=${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}

    echo "Running seed=$seed on GPU=$gpu"

    python train_link_prediction.py \
        --start_runs $seed \
        --end_runs $next_seed \
        --gpu $gpu \
        --model_name $model \
        --num_epochs 100 \
        --dataset_name $dataset &
}
done

wait


# 训练节点分类任务

for i in "${!seeds[@]}"; do
{
    seed=${seeds[$i]}
    next_seed=$((seed + 1))

    # 从 GPU 列表循环选择 GPU
    gpu=${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}

    echo "Running seed=$seed on GPU=$gpu"

    python train_gt.py \
        --start_runs $seed \
        --end_runs $next_seed \
        --gpu $gpu \
        --model_name $model \
        --num_epochs 200 \
        --dataset_name $dataset &
}
done

wait
