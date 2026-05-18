#!/bin/bash

model_name=GraphMixer
decoder_name=mlp
dataset=dsub1m
date=1204
gt_weight=0.5
alpha=0.1
method1=npl
method2=ptcl_2d
gpus=(0 0 0 1 1)

# Function to run the experiments and calculate the average
run_experiment() {
  local method=$1
  local prefix="${date}_${method}_${model_name}_${decoder_name}_${dataset}_${gt_weight}_${alpha}"

  # Create or clear the result file
  result_file="results/${prefix}_results.txt"
  echo "AUC   ACC" > $result_file

  for seed in 0 1 2 3 4
  do  
    {
      next_seed=$((seed + 1))
      gpu_id=${gpus[$seed]}

      read result1 result2 < <(python train.py \
            --prefix $prefix \
            --method $method \
            --start_runs $seed \
            --end_runs $next_seed \
            --gpu $gpu_id \
            --mmodel_name $model_name \
            --emodel_name $decoder_name \
            --use_ps_back 1 \
            --alpha $alpha \
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
            --dataset_name $dataset)
      
      echo "$result1 $result2" >> $result_file
    } &
  done

  wait
}

run_experiment $method1
run_experiment $method2

