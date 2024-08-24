python train_ncem.py \
    --prefix 0823_warmup_EM_1d_tgat_red_0.9_35\
    --emodel_name TGAT\
    --decoder 1\
    --start_runs 3\
    --end_runs 5\
    --warmup_e_train 0\
    --warmup_m_train 0\
    --gt_weight 0.9\
    --dataset_name reddit\
    --gpu 3\

