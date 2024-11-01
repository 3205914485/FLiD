from collections import deque
import torch
import torch.nn as nn
import logging
import time
import sys
import os
import numpy as np
import warnings
import shutil
import torch
import torch.nn as nn

from utils.utils import set_random_seed
from utils.utils import get_neighbor_sampler
from utils.DataLoader import get_idx_data_loader, get_NcEM_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_node_classification_direct_args

from NcEM.Direct_init import Direct_init
from NcEM.EM_warmup import em_warmup
from NcEM.Direct import Direct
from NcEM.utils import log_and_save_metrics, log_average_metrics, save_results, update_pseudo_labels

cpu_num = 2
os.environ["OMP_NUM_THREADS"] = str(cpu_num)  # noqa
os.environ["MKL_NUM_THREADS"] = str(cpu_num)  # noqa
torch.set_num_threads(cpu_num)

if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    double_way_datasets = ['bot','bot22','dgraph','dsub','yelp']
    # get arguments
    args = get_node_classification_direct_args()
    # get data for training, validation and testing
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, num_interactions, \
        num_node_features, val_offest, test_offest, train_nodes, num_classes, ps_batch_mask = \
        get_NcEM_data(dataset_name=args.dataset_name, val_ratio=args.val_ratio,
                      test_ratio=args.test_ratio, new_spilt=args.new_spilt, em_patience=args.iter_patience)
    args.num_classes = num_classes
    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)

    # get data loaders
    full_idx_data_loader = get_idx_data_loader(indices_list=list(
        range(len(full_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    train_idx_data_loader = get_idx_data_loader(indices_list=list(
        range(len(train_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    val_idx_data_loader = get_idx_data_loader(indices_list=list(
        range(len(val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(
        range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)

    data = {
        "node_raw_features": node_raw_features,
        "edge_raw_features": edge_raw_features,
        "full_data": full_data,
        "train_data": train_data,
        "train_nodes": train_nodes, # for transductive
        "val_data": val_data,
        "val_offest": val_offest,
        "test_data": test_data,
        "test_offest": test_offest,
        "full_neighbor_sampler": full_neighbor_sampler,
        "full_idx_data_loader": full_idx_data_loader,
        "train_idx_data_loader": train_idx_data_loader,
        "val_idx_data_loader": val_idx_data_loader,
        "test_idx_data_loader": test_idx_data_loader,
        "dataset_name": args.dataset_name,
        "ps_batch_mask": ps_batch_mask
    }

    Direct_metric_all_runs, Direct_metric_all_runs = [], []

    for run in range(args.start_runs, args.end_runs):

        set_random_seed(seed=run)

        args.seed = run

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(
            f"./logs/Direct/{args.prefix}/{args.dataset_name}/seed_{args.seed}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(
            f"./logs/Direct/{args.prefix}/{args.dataset_name}/seed_{args.seed}/{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {args}')

        # NcEM strating:

        # EM data:
        pseudo_labels_save_path = f"processed_data/{args.dataset_name}/Direct/pseudo_labels/{args.emodel_name}/{args.seed}/"
        pseudo_entropy = deque(maxlen=args.pseudo_entropy_ws)

        if args.dataset_name in double_way_datasets:
            pseudo_labels = torch.zeros(
                2, num_interactions, device=args.device)
        else:
            pseudo_labels = torch.zeros(
                1, num_interactions, device=args.device)

        base_val_metric_dict, base_test_metric_dict, Direct_metric_dict, Direct_metric_dict = {}, {}, {}, {}
       
        # EM Warmup
        Dirtrainer = Direct_init(args=args,
                                     logger=logger,
                                     train_data=train_data,
                                     node_raw_features=node_raw_features,
                                     edge_raw_features=edge_raw_features,
                                     full_neighbor_sampler=full_neighbor_sampler
                                     )

        {
            # base_val_total_loss, base_val_metrics, base_test_total_loss, base_test_metrics = \
            #     em_warmup(args=args,
            #               data=data,
            #               logger=logger,
            #               Etrainer=Etrainer,
            #               Mtrainer=Mtrainer,
            #               pseudo_labels=pseudo_labels,
            #               pseudo_entropy=pseudo_entropy,
            #               src_node_embeddings=src_node_embeddings,
            #               dst_node_embeddings=dst_node_embeddings)
            
            # pseudo_labels, num_targets = update_pseudo_labels(
            #     data=data, pseudo_labels=pseudo_labels, pseudo_entropy=pseudo_entropy, threshold=args.pseudo_entropy_th, \
            #     use_ps_back=args.use_ps_back, double_way_dataset=double_way_datasets, use_transductive=args.use_transductive, em_patience=args.em_patience)

            # if Etrainer.model_name not in ['JODIE', 'DyRep', 'TGN']:
            #     log_and_save_metrics(logger, 'Warm-up base', base_val_total_loss,
            #                          base_val_metrics, base_val_metric_dict, 'validate')
            # log_and_save_metrics(logger, 'Warm-up base', base_test_total_loss,
            #                      base_test_metrics, base_test_metric_dict, 'test')

            # if args.warmup_e_train or args.warmup_m_train:
            #     if run < args.end_runs - 1:
            #         logger.removeHandler(fh)
            #         logger.removeHandler(ch)
            #     continue

            # EM training
        }
        model_name = Dirtrainer.model_name
        save_model_name = f'Direct_{model_name}'
        save_model_folder = f"./saved_models/Direct/whole/{args.prefix}/{args.dataset_name}/{args.seed}/{save_model_name}/"
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)
        early_stopping = EarlyStopping(patience=args.iter_patience, save_model_folder=save_model_folder,
                                       save_model_name=save_model_name, logger=logger, model_name=model_name)

        IterDirect_metric_dict, IterDirect_metric_dict= {}, {}
        best_test_all = [0.0,0.0]
        for k in range(args.num_iters):
            logger.info(f'Direct train Iter {k + 1} starts.\n')
            if args.gt_weight != 1.0 and k != 0:
                gt_weight = 0.1 + (args.gt_weight - 0.1) * np.exp(-0.1 * k)
            else:
                gt_weight = 1.0
            Direct_total_Loss, Direct_metrics, Direct_total_loss, Direct_metrics = \
                Direct(args=args, gt_weight=gt_weight, data=data, logger=logger, Dirtrainer=Dirtrainer, 
                pseudo_labels=pseudo_labels, pseudo_entropy=pseudo_entropy)

            pseudo_labels, num_targets = update_pseudo_labels(
                data=data, pseudo_labels=pseudo_labels, pseudo_entropy=pseudo_entropy, threshold=args.pseudo_entropy_th, save_path=pseudo_labels_save_path, \
                use_ps_back=args.use_ps_back, double_way_dataset=double_way_datasets, use_transductive=args.use_transductive,save=args.save_pseudo_labels, iter_num=k, em_patience=args.iter_patience)

            logger.info(f"Iter: {k+1}, The sliding windows has {num_targets} sets entropy")
            
            if Dirtrainer.model_name not in ['JODIE', 'DyRep', 'TGN']:
                log_and_save_metrics(
                    logger, 'Direct', Direct_total_Loss, Direct_metrics, Direct_metric_dict, 'validate')
            log_and_save_metrics(
                logger, 'Direct', Direct_total_loss, Direct_metrics, Direct_metric_dict, 'test')

            if list(Direct_metrics.values())[0] > best_test_all[0]:
                best_test_all = list(Direct_metrics.values())
                if Dirtrainer.model_name not in ['JODIE', 'DyRep', 'TGN']:
                    IterDirect_metric_dict = Direct_metric_dict
                IterDirect_metric_dict = Direct_metric_dict

            logger.info(f'Best iter metrics, auc: {best_test_all[0]}, acc: {best_test_all[1]},')

            test_metric_indicator = []
            for metric_name in Direct_metrics.keys():
                test_metric_indicator.append(
                    (metric_name, Direct_metrics[metric_name], True))
            early_stop = early_stopping.step(
                test_metric_indicator, Dirtrainer.model)

            if early_stop[0]:
                break

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        # avoid the overlap of logs
        if run < args.end_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)
    
    print(f"{best_test_all[0]:.4f} {best_test_all[1]:.4f}")
    sys.exit()
