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
from utils.DataLoader import get_idx_data_loader, get_PTCL_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_node_classification_em_args

from PTCL.EM_init import em_init
from PTCL.EM_warmup import em_warmup
from PTCL.E_step import e_step
from PTCL.M_step import m_step
from PTCL.utils import log_and_save_metrics, save_results, update_pseudo_labels

from SEM.E_step import sem_e_step
from SEM.M_step import sem_m_step

from NPL.NPL import NPL_train
from NPL.NPL_init import NPL_init

cpu_num = 2
os.environ["OMP_NUM_THREADS"] = str(cpu_num)  # noqa
os.environ["MKL_NUM_THREADS"] = str(cpu_num)  # noqa
torch.set_num_threads(cpu_num)

def PTCL(args, data):
    
    for run in range(args.start_runs, args.end_runs):

        set_random_seed(seed=run)

        args.seed = run

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(
            f"./logs/ptcl/{args.prefix}/{args.dataset_name}/seed_{args.seed}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(
            f"./logs/ptcl/{args.prefix}/{args.dataset_name}/seed_{args.seed}/{str(time.time())}.log")
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

        # PTCL strating:

        # EM data:
        pseudo_labels_save_path = f"processed_data/{args.dataset_name}/pseudo_labels/{args.emodel_name}/{args.seed}/"

        src_node_embeddings = torch.zeros(
            num_interactions, num_node_features, device=args.device)
        dst_node_embeddings = torch.zeros(
            num_interactions, num_node_features, device=args.device)

        if args.dataset_name in args.double_way_datasets:
            pseudo_labels = torch.zeros(
                2, num_interactions, device=args.device)
        else:
            pseudo_labels = torch.zeros(
                1, num_interactions, device=args.device)

        pseudo_labels_store = []
            
        base_val_metric_dict, base_test_metric_dict, Eval_metric_dict, Etest_metric_dict, Mval_metric_dict, Mtest_metric_dict = {}, {}, {}, {}, {}, {}
       
        # EM Warmup
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
                      pseudo_labels_store=pseudo_labels_store,
                      src_node_embeddings=src_node_embeddings,
                      dst_node_embeddings=dst_node_embeddings)

        if args.decoder == 2:
            Mtrainer.model[0].load_state_dict(Mtrainer.model[1].state_dict())

        pseudo_labels = update_pseudo_labels(
            data=data, pseudo_labels=pseudo_labels, pseudo_labels_store=pseudo_labels_store, mode=args.mode, ps_filter=args.ps_filter,\
            double_way_dataset=args.double_way_datasets, use_transductive=args.use_transductive, threshold=args.filter_threshold)

        if Etrainer.model_name not in ['TGN']:
            log_and_save_metrics(logger, 'Warm-up base', base_val_total_loss,
                                 base_val_metrics, base_val_metric_dict, 'validate')
        log_and_save_metrics(logger, 'Warm-up base', base_test_total_loss,
                             base_test_metrics, base_test_metric_dict, 'test')

        if args.warmup_e_train or args.warmup_m_train:
            if run < args.end_runs - 1:
                logger.removeHandler(fh)
                logger.removeHandler(ch)
            continue

        # EM training
        model_name = Etrainer.model_name
        save_model_name = f'ptcl_{model_name}'
        save_model_folder = f"./saved_models/ptcl/EM/{args.prefix}/{args.dataset_name}/{args.seed}/{save_model_name}/"
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)
        early_stopping = EarlyStopping(patience=args.iter_patience, save_model_folder=save_model_folder,
                                       save_model_name=save_model_name, logger=logger, model_name=model_name)

        IterEval_metric_dict, IterEtest_metric_dict, IterMval_metric_dict, IterMtest_metric_dict = {}, {}, {}, {}
        best_test_all = [0.0,0.0]
        for k in range(args.num_em_iters):
            logger.info(f'E-M Iter {k + 1} starts.\n')
            if args.gt_weight != 1.0:
                gt_weight = 0.1 + (args.gt_weight - 0.1) * np.exp(-0.1 * k)
            else:
                gt_weight = 1.0

            Eval_total_loss, Eval_metrics, Etest_total_loss, Etest_metrics = \
                e_step(args=args, gt_weight=gt_weight, data=data, logger=logger, Etrainer=Etrainer, Mtrainer=Mtrainer, pseudo_labels=pseudo_labels,
                    src_node_embeddings=src_node_embeddings, dst_node_embeddings=dst_node_embeddings, iter_num=k)

            Mval_total_loss, Mval_metrics, Mtest_total_loss, Mtest_metrics = \
                m_step(args=args, data=data, logger=logger, Etrainer=Etrainer, Mtrainer=Mtrainer, pseudo_labels=pseudo_labels,
                       pseudo_labels_store=pseudo_labels_store, src_node_embeddings=src_node_embeddings, dst_node_embeddings=dst_node_embeddings)

            pseudo_labels = update_pseudo_labels(
                data=data, pseudo_labels=pseudo_labels, pseudo_labels_store=pseudo_labels_store, save_path=pseudo_labels_save_path, mode=args.mode, ps_filter=args.ps_filter,\
                double_way_dataset=args.double_way_datasets, use_transductive=args.use_transductive,save=args.save_pseudo_labels, iter_num=k, threshold=args.filter_threshold)
            
            if Etrainer.model_name not in ['TGN']:
                log_and_save_metrics(
                    logger, 'Estep', Eval_total_loss, Eval_metrics, Eval_metric_dict, 'validate')
                log_and_save_metrics(
                    logger, 'Mstep', Mval_total_loss, Mval_metrics, Mval_metric_dict, 'validate')
            log_and_save_metrics(
                logger, 'Estep', Etest_total_loss, Etest_metrics, Etest_metric_dict, 'test')
            log_and_save_metrics(
                logger, 'Mstep', Mtest_total_loss, Mtest_metrics, Mtest_metric_dict, 'test')

            if args.dataset_name in ['oag']:
                if list(Mtest_metrics.values())[1] > best_test_all[1]:
                    best_test_all = list(Mtest_metrics.values())
                    if Etrainer.model_name not in ['TGN']:
                        IterEval_metric_dict, IterMval_metric_dict = Eval_metric_dict, Mval_metric_dict
                    IterEtest_metric_dict, IterMtest_metric_dict = Etest_metric_dict, Mtest_metric_dict
            else:
                if list(Mtest_metrics.values())[0] > best_test_all[0]:
                    best_test_all = list(Mtest_metrics.values())
                    if Etrainer.model_name not in ['TGN']:
                        IterEval_metric_dict, IterMval_metric_dict = Eval_metric_dict, Mval_metric_dict
                    IterEtest_metric_dict, IterMtest_metric_dict = Etest_metric_dict, Mtest_metric_dict

            logger.info(f'Best iter metrics, auc: {best_test_all[0]}, acc: {best_test_all[1]}')

            test_metric_indicator = []
            for metric_name in Mtest_metrics.keys():
                test_metric_indicator.append(
                    (metric_name, Mtest_metrics[metric_name], True))
            early_stop = early_stopping.step(
                test_metric_indicator, nn.Sequential(Etrainer.model, Mtrainer.model), dataset_name=args.dataset_name)

            if early_stop[0]:
                break

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        # avoid the overlap of logs
        if run < args.end_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result
        save_results(args, Etrainer, IterEval_metric_dict, IterEtest_metric_dict, IterMval_metric_dict, IterMtest_metric_dict, run)

    return best_test_all

def SEM(args, data):
    
    for run in range(args.start_runs, args.end_runs):

        set_random_seed(seed=run)

        args.seed = run

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(
            f"./logs/sem/{args.prefix}/{args.dataset_name}/seed_{args.seed}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(
            f"./logs/sem/{args.prefix}/{args.dataset_name}/seed_{args.seed}/{str(time.time())}.log")
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

        # PTCL strating:

        # EM data:
        pseudo_labels_save_path = f"processed_data/{args.dataset_name}/pseudo_labels/{args.emodel_name}/{args.seed}/"

        src_node_embeddings = torch.zeros(
            num_interactions, num_node_features, device=args.device)
        dst_node_embeddings = torch.zeros(
            num_interactions, num_node_features, device=args.device)

        if args.dataset_name in args.double_way_datasets:
            pseudo_labels = torch.zeros(
                2, num_interactions, device=args.device)
        else:
            pseudo_labels = torch.zeros(
                1, num_interactions, device=args.device)

        pseudo_labels_store = []
            
        base_val_metric_dict, base_test_metric_dict, Eval_metric_dict, Etest_metric_dict, Mval_metric_dict, Mtest_metric_dict = {}, {}, {}, {}, {}, {}
       
        # EM Warmup
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
                      pseudo_labels_store=pseudo_labels_store,
                      src_node_embeddings=src_node_embeddings,
                      dst_node_embeddings=dst_node_embeddings)

        if args.decoder == 2:
            Mtrainer.model[0].load_state_dict(Mtrainer.model[1].state_dict())

        pseudo_labels = update_pseudo_labels(
            data=data, pseudo_labels=pseudo_labels, pseudo_labels_store=pseudo_labels_store, mode=args.mode, ps_filter=args.ps_filter,\
            double_way_dataset=args.double_way_datasets, use_transductive=args.use_transductive, threshold=args.filter_threshold)

        if Etrainer.model_name not in ['TGN']:
            log_and_save_metrics(logger, 'Warm-up base', base_val_total_loss,
                                 base_val_metrics, base_val_metric_dict, 'validate')
        log_and_save_metrics(logger, 'Warm-up base', base_test_total_loss,
                             base_test_metrics, base_test_metric_dict, 'test')

        if args.warmup_e_train or args.warmup_m_train:
            if run < args.end_runs - 1:
                logger.removeHandler(fh)
                logger.removeHandler(ch)
            continue

        # EM training
        model_name = Etrainer.model_name
        save_model_name = f'sem_{model_name}'
        save_model_folder = f"./saved_models/sem/EM/{args.prefix}/{args.dataset_name}/{args.seed}/{save_model_name}/"
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)
        early_stopping = EarlyStopping(patience=args.iter_patience, save_model_folder=save_model_folder,
                                       save_model_name=save_model_name, logger=logger, model_name=model_name)

        IterEval_metric_dict, IterEtest_metric_dict, IterMval_metric_dict, IterMtest_metric_dict = {}, {}, {}, {}
        best_test_all = [0.0,0.0]
        for k in range(args.num_em_iters):
            logger.info(f'E-M Iter {k + 1} starts.\n')
            if args.gt_weight != 1.0:
                gt_weight = 0.1 + (args.gt_weight - 0.1) * np.exp(-0.1 * k)
            else:
                gt_weight = 1.0

            Eval_total_loss, Eval_metrics, Etest_total_loss, Etest_metrics = \
                sem_e_step(args=args, gt_weight=gt_weight, data=data, logger=logger, Etrainer=Etrainer, Mtrainer=Mtrainer, pseudo_labels=pseudo_labels,
                    src_node_embeddings=src_node_embeddings, dst_node_embeddings=dst_node_embeddings, iter_num=k)
                
            pseudo_labels = update_pseudo_labels(
                data=data, pseudo_labels=pseudo_labels, pseudo_labels_store=pseudo_labels_store, save_path=pseudo_labels_save_path, mode=args.mode, ps_filter=args.ps_filter,\
                double_way_dataset=args.double_way_datasets, use_transductive=args.use_transductive,save=args.save_pseudo_labels, iter_num=k, threshold=args.filter_threshold)
                
            Mval_total_loss, Mval_metrics, Mtest_total_loss, Mtest_metrics = \
                sem_m_step(args=args, gt_weight=gt_weight, data=data, logger=logger, Etrainer=Etrainer, Mtrainer=Mtrainer, pseudo_labels=pseudo_labels,
                       pseudo_labels_store=pseudo_labels_store, src_node_embeddings=src_node_embeddings, dst_node_embeddings=dst_node_embeddings, iter_num=k)

            pseudo_labels = update_pseudo_labels(
                data=data, pseudo_labels=pseudo_labels, pseudo_labels_store=pseudo_labels_store, save_path=pseudo_labels_save_path, mode=args.mode, ps_filter=args.ps_filter,\
                double_way_dataset=args.double_way_datasets, use_transductive=args.use_transductive,save=args.save_pseudo_labels, iter_num=k, threshold=args.filter_threshold)
            
            if Etrainer.model_name not in ['TGN']:
                log_and_save_metrics(
                    logger, 'Estep', Eval_total_loss, Eval_metrics, Eval_metric_dict, 'validate')
                log_and_save_metrics(
                    logger, 'Mstep', Mval_total_loss, Mval_metrics, Mval_metric_dict, 'validate')
            log_and_save_metrics(
                logger, 'Estep', Etest_total_loss, Etest_metrics, Etest_metric_dict, 'test')
            log_and_save_metrics(
                logger, 'Mstep', Mtest_total_loss, Mtest_metrics, Mtest_metric_dict, 'test')

            if args.dataset_name in ['oag']:
                if list(Mtest_metrics.values())[1] > best_test_all[1]:
                    best_test_all = list(Mtest_metrics.values())
                    if Etrainer.model_name not in ['TGN']:
                        IterEval_metric_dict, IterMval_metric_dict = Eval_metric_dict, Mval_metric_dict
                    IterEtest_metric_dict, IterMtest_metric_dict = Etest_metric_dict, Mtest_metric_dict
            else:
                if list(Mtest_metrics.values())[0] > best_test_all[0]:
                    best_test_all = list(Mtest_metrics.values())
                    if Etrainer.model_name not in ['TGN']:
                        IterEval_metric_dict, IterMval_metric_dict = Eval_metric_dict, Mval_metric_dict
                    IterEtest_metric_dict, IterMtest_metric_dict = Etest_metric_dict, Mtest_metric_dict

            logger.info(f'Best iter metrics, auc: {best_test_all[0]}, acc: {best_test_all[1]}')

            test_metric_indicator = []
            for metric_name in Mtest_metrics.keys():
                test_metric_indicator.append(
                    (metric_name, Mtest_metrics[metric_name], True))
            early_stop = early_stopping.step(
                test_metric_indicator, nn.Sequential(Etrainer.model, Mtrainer.model), dataset_name=args.dataset_name)

            if early_stop[0]:
                break

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        # avoid the overlap of logs
        if run < args.end_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result
        save_results(args, Etrainer, IterEval_metric_dict, IterEtest_metric_dict, IterMval_metric_dict, IterMtest_metric_dict, run)

    return best_test_all

def NPL(args, data):

    for run in range(args.start_runs, args.end_runs):

        set_random_seed(seed=run)

        args.seed = run

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(
            f"./logs/npl/{args.prefix}/{args.dataset_name}/seed_{args.seed}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(
            f"./logs/npl/{args.prefix}/{args.dataset_name}/seed_{args.seed}/{str(time.time())}.log")
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

        # PTCL strating:

        # EM data:
        pseudo_labels_save_path = f"processed_data/{args.dataset_name}/npl/pseudo_labels/{args.emodel_name}/{args.seed}/"

        if args.dataset_name in args.double_way_datasets:
            pseudo_labels = torch.zeros(
                2, num_interactions, device=args.device)
        else:
            pseudo_labels = torch.zeros(
                1, num_interactions, device=args.device)
        pseudo_labels_store = []
        NPL_val_metric_dict, NPL_test_metric_dict = {}, {}
       
        # EM Warmup
        Dirtrainer = NPL_init(args=args,
                                     logger=logger,
                                     train_data=train_data,
                                     node_raw_features=node_raw_features,
                                     edge_raw_features=edge_raw_features,
                                     full_neighbor_sampler=full_neighbor_sampler
                                     )

        model_name = Dirtrainer.model_name
        save_model_name = f'NPL_{model_name}'
        save_model_folder = f"./saved_models/npl/whole/{args.prefix}/{args.dataset_name}/{args.seed}/{save_model_name}/"
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)
        early_stopping = EarlyStopping(patience=args.iter_patience, save_model_folder=save_model_folder,
                                       save_model_name=save_model_name, logger=logger, model_name=model_name)

        IterNPL_val_metric_dict, IterNPL_test_metric_dict= {}, {}
        best_test_all = [0.0,0.0]
        # update first to get the ground truth for pseudo labels
        pseudo_labels = update_pseudo_labels(
            data=data, pseudo_labels=pseudo_labels, pseudo_labels_store=pseudo_labels_store, mode=args.mode, ps_filter='none',\
            double_way_dataset=args.double_way_datasets, use_transductive=args.use_transductive, threshold=args.filter_threshold)
        # set the ps_filter to none cause no warmup for getting the pseudo_labels_store
        for k in range(args.num_iters):
            logger.info(f'NPL train Iter {k + 1} starts.\n')
            if args.gt_weight != 1.0 and k != 0:
                gt_weight = 0.1 + (args.gt_weight - 0.1) * np.exp(-args.alpha * k)
            else:
                gt_weight = 1.0

            NPL_val_Loss, NPL_val_metrics, NPL_test_loss, NPL_test_metrics = \
                NPL_train(args=args, gt_weight=gt_weight, data=data, logger=logger, Dirtrainer=Dirtrainer, iter_num=k,
                pseudo_labels=pseudo_labels)

            pseudo_labels = update_pseudo_labels(
                data=data, pseudo_labels=pseudo_labels, pseudo_labels_store=pseudo_labels_store, save_path=pseudo_labels_save_path, mode=args.mode, ps_filter=args.ps_filter,\
                double_way_dataset=args.double_way_datasets, use_transductive=args.use_transductive,save=args.save_pseudo_labels, iter_num=k, threshold=args.filter_threshold)
            
            if Dirtrainer.model_name not in ['TGN']:
                log_and_save_metrics(
                    logger, 'NPL', NPL_val_Loss, NPL_val_metrics, NPL_val_metric_dict, 'validate')
            log_and_save_metrics(
                logger, 'NPL', NPL_test_loss, NPL_test_metrics, NPL_test_metric_dict, 'test')

            if args.dataset_name in ['oag']:
                if list(NPL_test_metrics.values())[1] > best_test_all[1]:
                    best_test_all = list(NPL_test_metrics.values())
                    if Dirtrainer.model_name not in ['TGN']:
                        IterNPL_val_metric_dict = NPL_val_metric_dict
                    IterNPL_test_metric_dict = NPL_test_metric_dict
            else :
                if list(NPL_test_metrics.values())[0] > best_test_all[0]:
                    best_test_all = list(NPL_test_metrics.values())
                    if Dirtrainer.model_name not in ['TGN']:
                        IterNPL_val_metric_dict = NPL_val_metric_dict
                    IterNPL_test_metric_dict = NPL_test_metric_dict 

            logger.info(f'Best iter metrics, auc: {best_test_all[0]}, acc: {best_test_all[1]},')

            test_metric_indicator = []
            for metric_name in NPL_test_metrics.keys():
                test_metric_indicator.append(
                    (metric_name, NPL_test_metrics[metric_name], True))
            early_stop = early_stopping.step(
                test_metric_indicator, Dirtrainer.model, dataset_name=args.dataset_name)

            if early_stop[0]:
                break

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        # avoid the overlap of logs
        if run < args.end_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result
        save_results(args, Etrainer, [], [], IterNPL_val_metric_dict, IterNPL_test_metric_dict, run)
        # No E-step for NPL
    return best_test_all

def PTCL_2D(args, data):
    
    args.decoder = 2
    for run in range(args.start_runs, args.end_runs):

        set_random_seed(seed=run)

        args.seed = run

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(
            f"./logs/ptcl/{args.prefix}/{args.dataset_name}/seed_{args.seed}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(
            f"./logs/ptcl/{args.prefix}/{args.dataset_name}/seed_{args.seed}/{str(time.time())}.log")
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

        # PTCL strating:

        # EM data:
        pseudo_labels_save_path = f"processed_data/{args.dataset_name}/pseudo_labels/{args.emodel_name}/{args.seed}/"

        src_node_embeddings = torch.zeros(
            num_interactions, num_node_features, device=args.device)
        dst_node_embeddings = torch.zeros(
            num_interactions, num_node_features, device=args.device)

        if args.dataset_name in args.double_way_datasets:
            pseudo_labels = torch.zeros(
                2, num_interactions, device=args.device)
        else:
            pseudo_labels = torch.zeros(
                1, num_interactions, device=args.device)

        pseudo_labels_store = []
            
        base_val_metric_dict, base_test_metric_dict, Eval_metric_dict, Etest_metric_dict, Mval_metric_dict, Mtest_metric_dict = {}, {}, {}, {}, {}, {}
       
        # EM Warmup
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
                      pseudo_labels_store=pseudo_labels_store,
                      src_node_embeddings=src_node_embeddings,
                      dst_node_embeddings=dst_node_embeddings)

        if args.decoder == 2:
            Mtrainer.model[0].load_state_dict(Mtrainer.model[1].state_dict())

        pseudo_labels = update_pseudo_labels(
            data=data, pseudo_labels=pseudo_labels, pseudo_labels_store=pseudo_labels_store, mode=args.mode, ps_filter=args.ps_filter,\
            double_way_dataset=args.double_way_datasets, use_transductive=args.use_transductive, threshold=args.filter_threshold)

        if Etrainer.model_name not in ['TGN']:
            log_and_save_metrics(logger, 'Warm-up base', base_val_total_loss,
                                 base_val_metrics, base_val_metric_dict, 'validate')
        log_and_save_metrics(logger, 'Warm-up base', base_test_total_loss,
                             base_test_metrics, base_test_metric_dict, 'test')

        if args.warmup_e_train or args.warmup_m_train:
            if run < args.end_runs - 1:
                logger.removeHandler(fh)
                logger.removeHandler(ch)
            continue

        # EM training
        model_name = Etrainer.model_name
        save_model_name = f'ptcl_{model_name}'
        save_model_folder = f"./saved_models/ptcl/EM/{args.prefix}/{args.dataset_name}/{args.seed}/{save_model_name}/"
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)
        early_stopping = EarlyStopping(patience=args.iter_patience, save_model_folder=save_model_folder,
                                       save_model_name=save_model_name, logger=logger, model_name=model_name)

        IterEval_metric_dict, IterEtest_metric_dict, IterMval_metric_dict, IterMtest_metric_dict = {}, {}, {}, {}
        best_test_all = [0.0,0.0]
        for k in range(args.num_em_iters):
            logger.info(f'E-M Iter {k + 1} starts.\n')
            if args.gt_weight != 1.0:
                gt_weight = 0.1 + (args.gt_weight - 0.1) * np.exp(-0.1 * k)
            else:
                gt_weight = 1.0

            Eval_total_loss, Eval_metrics, Etest_total_loss, Etest_metrics = \
                e_step(args=args, gt_weight=gt_weight, data=data, logger=logger, Etrainer=Etrainer, Mtrainer=Mtrainer, pseudo_labels=pseudo_labels,
                    src_node_embeddings=src_node_embeddings, dst_node_embeddings=dst_node_embeddings, iter_num=k)

            Mval_total_loss, Mval_metrics, Mtest_total_loss, Mtest_metrics = \
                m_step(args=args, data=data, logger=logger, Etrainer=Etrainer, Mtrainer=Mtrainer, pseudo_labels=pseudo_labels,
                       pseudo_labels_store=pseudo_labels_store, src_node_embeddings=src_node_embeddings, dst_node_embeddings=dst_node_embeddings)

            pseudo_labels = update_pseudo_labels(
                data=data, pseudo_labels=pseudo_labels, pseudo_labels_store=pseudo_labels_store, save_path=pseudo_labels_save_path, mode=args.mode, ps_filter=args.ps_filter,\
                double_way_dataset=args.double_way_datasets, use_transductive=args.use_transductive,save=args.save_pseudo_labels, iter_num=k, threshold=args.filter_threshold)
            
            if Etrainer.model_name not in ['TGN']:
                log_and_save_metrics(
                    logger, 'Estep', Eval_total_loss, Eval_metrics, Eval_metric_dict, 'validate')
                log_and_save_metrics(
                    logger, 'Mstep', Mval_total_loss, Mval_metrics, Mval_metric_dict, 'validate')
            log_and_save_metrics(
                logger, 'Estep', Etest_total_loss, Etest_metrics, Etest_metric_dict, 'test')
            log_and_save_metrics(
                logger, 'Mstep', Mtest_total_loss, Mtest_metrics, Mtest_metric_dict, 'test')

            if args.dataset_name in ['oag']:
                if list(Mtest_metrics.values())[1] > best_test_all[1]:
                    best_test_all = list(Mtest_metrics.values())
                    if Etrainer.model_name not in ['TGN']:
                        IterEval_metric_dict, IterMval_metric_dict = Eval_metric_dict, Mval_metric_dict
                    IterEtest_metric_dict, IterMtest_metric_dict = Etest_metric_dict, Mtest_metric_dict
            else:
                if list(Mtest_metrics.values())[0] > best_test_all[0]:
                    best_test_all = list(Mtest_metrics.values())
                    if Etrainer.model_name not in ['TGN']:
                        IterEval_metric_dict, IterMval_metric_dict = Eval_metric_dict, Mval_metric_dict
                    IterEtest_metric_dict, IterMtest_metric_dict = Etest_metric_dict, Mtest_metric_dict

            logger.info(f'Best iter metrics, auc: {best_test_all[0]}, acc: {best_test_all[1]}')

            test_metric_indicator = []
            for metric_name in Mtest_metrics.keys():
                test_metric_indicator.append(
                    (metric_name, Mtest_metrics[metric_name], True))
            early_stop = early_stopping.step(
                test_metric_indicator, nn.Sequential(Etrainer.model, Mtrainer.model), dataset_name=args.dataset_name)

            if early_stop[0]:
                break

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        # avoid the overlap of logs
        if run < args.end_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result
        save_results(args, Etrainer, IterEval_metric_dict, IterEtest_metric_dict, IterMval_metric_dict, IterMtest_metric_dict, run)

    return best_test_all

if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    # get arguments
    args = get_node_classification_em_args()
    # get data for training, validation and testing
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, num_interactions, \
        num_node_features, val_offest, test_offest, train_nodes, test_nodes, num_classes, ps_batch_mask = \
        get_PTCL_data(dataset_name=args.dataset_name, val_ratio=args.val_ratio,
                      test_ratio=args.test_ratio, new_spilt=args.new_spilt, iter_patience=args.iter_patience)
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
        "test_nodes": test_nodes,
        "full_neighbor_sampler": full_neighbor_sampler,
        "full_idx_data_loader": full_idx_data_loader,
        "train_idx_data_loader": train_idx_data_loader,
        "val_idx_data_loader": val_idx_data_loader,
        "test_idx_data_loader": test_idx_data_loader,
        "dataset_name": args.dataset_name,
        "ps_batch_mask": ps_batch_mask
    }

    if args.method == 'PTCL':
        best_test_all = PTCL(args, data)
    elif args.method == 'SEM':
        best_test_all = SEM(args, data)
    elif args.method == 'NPL':
        best_test_all = NPL(args, data)
    elif args.method == 'PTCL-2D':
        best_test_all = PTCL_2D(args, data)
    else:
        raise ValueError(f"Wrong value for method {args.method}!")

    print(f"{best_test_all[0]:.4f} {best_test_all[1]:.4f}")
    sys.exit()