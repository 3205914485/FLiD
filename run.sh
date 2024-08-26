python train_ncem.py \
    --prefix 0826_EM_1d_tgn_red_0.8_02\
    --emodel_name TGN\
    --use_confidence 0\
    --confidence_threshold 0.5\
    --decoder 1\
    --start_runs 0\
    --end_runs 2\
    --warmup_e_train 0\
    --warmup_m_train 0\
    --gt_weight 0.8\
    --dataset_name reddit\
    --gpu 3\

