python train_ncem.py \
    --prefix 0827_LT_1d_tgn_red_0.9_45\
    --start_runs 4\
    --end_runs 5\
    --gpu 1\
    --emodel_name TGN\
    --use_confidence 0\
    --confidence_threshold 0.5\
    --decoder 2\
    --num_epochs_m_step 40\
    --num_epochs_e_step 25\
    --mw_patience 10\
    --em_patience 5\
    --patience 10\
    --warmup_e_train 0\
    --warmup_m_train 0\
    --gt_weight 0.9\
    --dataset_name reddit\

