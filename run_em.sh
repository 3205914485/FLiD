python train_ncem.py \
    --prefix 0826_LT_1d_tgn_wiki_0.9_02\
    --start_runs 0\
    --end_runs 2\
    --gpu 1\
    --emodel_name TGN\
    --use_confidence 0\
    --confidence_threshold 0.5\
    --decoder 1\
    --num_epochs_m_step 50\
    --mw_patience 20\
    --warmup_e_train 0\
    --warmup_m_train 0\
    --gt_weight 0.9\
    --dataset_name wikipedia\

