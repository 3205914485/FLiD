#!/bin/bash

model_name=TGAT
decoder_name=mlp
gpu=3
dataset=dsub1m
date=1225
gt_weight1=0.1

# Function to run the experiments and calculate the average
run_experiment() {
  local gt_weight=$1
  local prefix="${date}_warmup_${model_name}_${decoder_name}_${dataset}"

  # Create or clear the result file
  result_file="results/${prefix}_results.txt"
  echo "AUC   ACC" > $result_file

  for seed in 4
  do  
    {
      next_seed=$((seed + 1))
      read result1 result2 < <(python train.py \
            --prefix $prefix \
            --start_runs $seed \
            --end_runs $next_seed \
            --gpu $gpu \
            --mmodel_name $model_name \
            --emodel_name $decoder_name \
            --use_ps_back 0 \
            --decoder 1 \
            --num_em_iters 30\
            --num_epochs_m_step 50 \
            --num_epochs_e_step 50 \
            --num_epochs_e_warmup 100 \
            --num_epochs_m_warmup 100 \
            --iter_patience 5 \
            --patience 15 \
            --warmup_e_train 1 \
            --warmup_m_train 0 \
            --gt_weight $gt_weight \
            --dataset_name $dataset)
      
      echo "$result1 $result2" >> $result_file
    } &
  done

  # Wait for all background processes to complete
  wait

}
run_experiment $gt_weight1