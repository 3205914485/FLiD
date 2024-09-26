import copy
from datetime import datetime
from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
import time
import sys
import os
from tqdm import tqdm
import numpy as np
import warnings
import shutil
import torch
import torch.nn as nn
import json
from collections import deque

from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from utils.utils import get_neighbor_sampler
from evaluate_models_utils import evaluate_model_node_classification
from utils.DataLoader import get_idx_data_loader, get_node_classification_data, get_link_prediction_data, get_NcEM_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_node_classification_em_args

from NcEM.EM_init import em_init
from NcEM.EM_warmup import em_warmup
from NcEM.E_step import e_step
from NcEM.M_step import m_step
from NcEM.utils import log_and_save_metrics, log_average_metrics, save_results, update_pseudo_labels

cpu_num = 2
os.environ["OMP_NUM_THREADS"] = str(cpu_num)  # noqa
os.environ["MKL_NUM_THREADS"] = str(cpu_num)  # noqa
torch.set_num_threads(cpu_num)

if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    double_way_datasets = ['bot', 'bot22', 'dsub', 'dgraph']
    # get arguments
    args = get_node_classification_em_args()
    # get data for training, validation and testing
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, num_interactions, num_node_features, val_offest, test_offest, train_nodes = \
        get_NcEM_data(dataset_name=args.dataset_name, val_ratio=args.val_ratio,
                      test_ratio=args.test_ratio, new_spilt=args.new_spilt)

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
        "dataset_name": args.dataset_name
    }

    Eval_metric_all_runs, Etest_metric_all_runs = [], [],
    for run in range(args.start_runs, args.end_runs):

        set_random_seed(seed=run)

        args.seed = run

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(
            f"./logs/ncem/{args.prefix}/{args.dataset_name}/seed_{args.seed}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(
            f"./logs/ncem/{args.prefix}/{args.dataset_name}/seed_{args.seed}/{str(time.time())}.log")
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
        src_node_embeddings = torch.zeros(
            num_interactions, num_node_features, device=args.device)
        dst_node_embeddings = torch.zeros(
            num_interactions, num_node_features, device=args.device)
        if args.dataset_name in double_way_datasets:
            pseudo_labels = torch.zeros(
                2, num_interactions, device=args.device)
            pseudo_labels_confidence = []
        else:
            pseudo_labels = torch.zeros(
                1, num_interactions, device=args.device)
            pseudo_labels_confidence = []
        pseudo_entropy = deque(maxlen=args.pseudo_entropy_ws)
        base_val_metric_dict, base_test_metric_dict, Eval_metric_dict, Etest_metric_dict = {}, {}, {}, {}

        Etrainer, Mtrainer = em_init(args=args,
                                     logger=logger,
                                     train_data=train_data,
                                     node_raw_features=node_raw_features,
                                     edge_raw_features=edge_raw_features,
                                     full_neighbor_sampler=full_neighbor_sampler
                                     )

        base_val_total_loss, base_val_metrics, base_test_total_loss, base_test_metrics = \
            em_warmup(args=args,
                      data=data,
                      logger=logger,
                      Etrainer=Etrainer,
                      Mtrainer=Mtrainer,
                      pseudo_labels=pseudo_labels,
                      pseudo_entropy=pseudo_entropy,
                      src_node_embeddings=src_node_embeddings,
                      dst_node_embeddings=dst_node_embeddings)
        
        pseudo_labels, num_targets = update_pseudo_labels(
            data=data, pseudo_labels=pseudo_labels, pseudo_entropy=pseudo_entropy, threshold=args.pseudo_entropy_th, use_pseudo_entropy=args.use_entropy,double_way_dataset=double_way_datasets)

        if Etrainer.model_name not in ['JODIE', 'DyRep', 'TGN']:
            log_and_save_metrics(logger, 'Warm-up base', base_val_total_loss,
                                 base_val_metrics, base_val_metric_dict, 'validate')
        log_and_save_metrics(logger, 'Warm-up base', base_test_total_loss,
                             base_test_metrics, base_test_metric_dict, 'test')

        if args.warmup_e_train or args.warmup_m_train:
            if run < args.end_runs - 1:
                logger.removeHandler(fh)
                logger.removeHandler(ch)
            continue

        model_name = Etrainer.model_name

        IterEval_metric_dict, IterEtest_metric_dict, IterMval_metric_dict, IterMtest_metric_dict = {}, {}, {}, {}

        logger.info(f'E Unified train starts.\n')
        gt_weight = 1.0
        Eval_total_loss, Eval_metrics, Etest_total_loss, Etest_metrics = \
            e_step(args=args, gt_weight=gt_weight, data=data, logger=logger, Etrainer=Etrainer, Mtrainer=Mtrainer, pseudo_labels=pseudo_labels,
                    src_node_embeddings=src_node_embeddings, dst_node_embeddings=dst_node_embeddings)
        if Etrainer.model_name not in ['JODIE', 'DyRep', 'TGN']:
            log_and_save_metrics(
                logger, 'Estep', Eval_total_loss, Eval_metrics, Eval_metric_dict, 'validate')
        log_and_save_metrics(logger, 'Estep', Etest_total_loss,
                             Etest_metrics, Etest_metric_dict, 'test')

        if Etrainer.model_name not in ['JODIE', 'DyRep', 'TGN']:
            IterEval_metric_dict = Eval_metric_dict
        IterEtest_metric_dict = Etest_metric_dict

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        if Etrainer.model_name not in ['JODIE', 'DyRep', 'TGN']:
            Eval_metric_all_runs.append(IterEval_metric_dict)
        Etest_metric_all_runs.append(IterEtest_metric_dict)

        # avoid the overlap of logs
        if run < args.end_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)
    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.end_runs} runs:')

    if Etrainer.model_name not in ['JODIE', 'DyRep', 'TGN']:
        log_average_metrics(logger, Eval_metric_all_runs, 'Estep validate')

    log_average_metrics(logger, Etest_metric_all_runs, 'Estep test')

    sys.exit()
