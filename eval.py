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
from utils.DataLoader import get_idx_data_loader, get_PTCL_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_node_classification_em_args
from utils.metrics import get_node_classification_metrics_em
from tqdm import tqdm

from PTCL.EM_init import em_init
from PTCL.utils import log_and_save_metrics, save_results, update_pseudo_labels
from PTCL.E_step import evaluate_model_node_classification_withembeddings

cpu_num = 2
os.environ["OMP_NUM_THREADS"] = str(cpu_num)  # noqa
os.environ["MKL_NUM_THREADS"] = str(cpu_num)  # noqa
torch.set_num_threads(cpu_num)

if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    # get arguments
    args = get_node_classification_em_args()
    double_way_datasets = args.double_way_datasets
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

    Eval_metric_all_runs, Etest_metric_all_runs, Mval_metric_all_runs, Mtest_metric_all_runs = [], [], [], []

    for run in range(args.start_runs, args.end_runs):

        set_random_seed(seed=run)

        args.seed = run

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(
            f"./logs/eval/{args.method}/{args.prefix}/{args.dataset_name}/seed_{args.seed}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(
            f"./logs/eval/{args.method}/{args.prefix}/{args.dataset_name}/seed_{args.seed}/{str(time.time())}.log")
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

        # EM data:

        src_node_embeddings = torch.zeros(
            num_interactions, num_node_features, device=args.device)
        dst_node_embeddings = torch.zeros(
            num_interactions, num_node_features, device=args.device)

        if args.dataset_name in double_way_datasets:
            pseudo_labels = torch.zeros(
                2, num_interactions, device=args.device)
        else:
            pseudo_labels = torch.zeros(
                1, num_interactions, device=args.device)

        Mtrainer, Etrainer = em_init(args=args,
                                     logger=logger,
                                     train_data=train_data,
                                     node_raw_features=node_raw_features,
                                     edge_raw_features=edge_raw_features,
                                     full_neighbor_sampler=full_neighbor_sampler
                                     )

        model_name = Mtrainer.model_name
        save_model_name = f'{args.method}_{model_name}'
        save_model_folder = f"./saved_models/{args.method}/EM/{args.prefix}/{args.dataset_name}/{args.seed}/{save_model_name}/"
        early_stopping = EarlyStopping(patience=args.iter_patience, save_model_folder=save_model_folder,
                                       save_model_name=save_model_name, logger=logger, model_name=model_name)
        model = nn.Sequential(Mtrainer.model, Etrainer.model)
        early_stopping.load_checkpoint(model,map_location=args.device)
        loss_func = Etrainer.criterion
        model[0].eval()

        src_node_embeddings_list, dst_node_embeddings_list, pseudo_labels_list = [], [], []
        if model_name in ['TGAT', 'TGN', 'TCL', 'GraphMixer', 'DyGFormer']:
            model[0].set_neighbor_sampler(full_neighbor_sampler)
        if model_name in ['TGN']:
            model[0].memory_bank.__init_memory_bank__()

        idx_data_loader_tqdm = tqdm(full_idx_data_loader, ncols=120)
        for batch_idx, data_indices in enumerate(idx_data_loader_tqdm):
            data_indices = data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                full_data.src_node_ids[data_indices], full_data.dst_node_ids[data_indices], full_data.node_interact_times[data_indices], \
                full_data.edge_ids[data_indices]

            with torch.no_grad():
                if model_name in ['TGAT', 'TCL']:
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                        dst_node_ids=batch_dst_node_ids,
                                                                        node_interact_times=batch_node_interact_times,
                                                                        num_neighbors=args.num_neighbors)
                elif model_name in ['TGN']:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                        dst_node_ids=batch_dst_node_ids,
                                                                        node_interact_times=batch_node_interact_times,
                                                                        edge_ids=batch_edge_ids,
                                                                        edges_are_positive=True,
                                                                        num_neighbors=args.num_neighbors)
                elif model_name in ['GraphMixer']:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                        dst_node_ids=batch_dst_node_ids,
                                                                        node_interact_times=batch_node_interact_times,
                                                                        num_neighbors=args.num_neighbors,
                                                                        time_gap=args.time_gap)
                elif model_name in ['DyGFormer']:
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                        dst_node_ids=batch_dst_node_ids,
                                                                        node_interact_times=batch_node_interact_times)
                else:
                    raise ValueError(f"Wrong value for model_name {model_name}!")
            src_node_embeddings_list.append(batch_src_node_embeddings)
            dst_node_embeddings_list.append(batch_dst_node_embeddings)

        # Concatenate all embeddings
        new_src_embeddings = torch.cat(src_node_embeddings_list, dim=0)
        new_dst_embeddings = torch.cat(dst_node_embeddings_list, dim=0)
        src_node_embeddings.copy_(new_src_embeddings)
        dst_node_embeddings.copy_(new_dst_embeddings)

        best_metrics, best_epoch = {'roc_auc': 0.0, 'acc': 0.0}, 0

        train_total_loss, train_y_trues, train_y_predicts, batch_count = 0.0, [], [], 0.0
        for batch_idx, train_data_indices in enumerate(train_idx_data_loader):
            train_data_indices = train_data_indices.numpy()
            if args.dataset_name in double_way_datasets:
                batch_node_interact_times, batch_edge_ids, batch_labels, batch_labels_times = \
                    train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices], \
                    [train_data.labels[0][train_data_indices], train_data.labels[1][train_data_indices]], \
                    [train_data.labels_time[0][train_data_indices],
                        train_data.labels_time[1][train_data_indices]]
            else:
                batch_node_interact_times, batch_edge_ids, batch_labels, batch_labels_times = \
                    train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices], \
                    train_data.labels[train_data_indices], train_data.labels_time[train_data_indices]

            batch_src_node_embeddings, batch_dst_node_embeddings = \
                src_node_embeddings[batch_edge_ids -1], dst_node_embeddings[batch_edge_ids-1]

            if args.dataset_name in double_way_datasets:
                predicts = model[1][1](x=torch.cat(
                    [batch_src_node_embeddings, batch_dst_node_embeddings], dim=0))
                labels = torch.from_numpy(np.concatenate(
                    [batch_labels[0], batch_labels[1]], axis=0)).to(torch.long).to(predicts.device)
                if args.dataset_name == 'dsub':
                    mask_gt_src = torch.from_numpy(
                        (batch_node_interact_times == batch_labels_times[0]) & (np.isin(batch_labels[0],[0,1]))).to(torch.bool)
                    mask_gt_dst = torch.from_numpy(
                        (batch_node_interact_times == batch_labels_times[1]) & (np.isin(batch_labels[1],[0,1]))).to(torch.bool)
                else :
                    mask_gt_src = torch.from_numpy(
                        (batch_node_interact_times == batch_labels_times[0])).to(torch.bool)
                    mask_gt_dst = torch.from_numpy(
                        (batch_node_interact_times == batch_labels_times[1])).to(torch.bool)
                mask = torch.cat(
                    [mask_gt_src, mask_gt_dst], dim=0).squeeze(dim=-1)
                probabilities = torch.softmax(predicts, dim=1)
            else:
                predicts = model[1][1](x=batch_src_node_embeddings)
                labels = torch.from_numpy(batch_labels).to(
                    torch.long).to(predicts.device)
                mask = torch.from_numpy(
                    batch_node_interact_times == batch_labels_times).to(torch.bool)

            filtered_predicts = predicts[mask]
            filtered_labels = labels[mask]

            if filtered_predicts.size(0) > 0:
                loss = loss_func(input=filtered_predicts,
                                    target=filtered_labels).mean()

                train_total_loss += loss.item()

                train_y_trues.append(filtered_labels)
                train_y_predicts.append(filtered_predicts)
                batch_count += 1
        train_total_loss /= (batch_count)
        train_y_trues = torch.cat(train_y_trues, dim=0)
        train_y_predicts = torch.cat(train_y_predicts, dim=0)

        train_metrics = get_node_classification_metrics_em(
            predicts=train_y_predicts, labels=train_y_trues)

        val_total_loss, val_metrics = evaluate_model_node_classification_withembeddings(model=model[1][1],
                                                                                        dataset=args.dataset_name,
                                                                                        src_node_embeddings=src_node_embeddings,
                                                                                        dst_node_embeddings=dst_node_embeddings,
                                                                                        evaluate_idx_data_loader=val_idx_data_loader,
                                                                                        evaluate_data=val_data,
                                                                                        double_way_datasets=double_way_datasets,
                                                                                        loss_func=loss_func)
        for metric_name in train_metrics.keys():
            logger.info(
                f'train {metric_name}, {train_metrics[metric_name]:.4f}')
        logger.info(f'validate loss: {val_total_loss:.4f}')
        for metric_name in val_metrics.keys():
            logger.info(
                f'validate {metric_name}, {val_metrics[metric_name]:.4f}')

        # perform testing once after test_interval_epochs
        test_total_loss, test_metrics = evaluate_model_node_classification_withembeddings(model=model[1][1],
                                                                                            dataset=args.dataset_name,
                                                                                            src_node_embeddings=src_node_embeddings,
                                                                                            dst_node_embeddings=dst_node_embeddings,
                                                                                            evaluate_idx_data_loader=test_idx_data_loader,
                                                                                            evaluate_data=test_data,
                                                                                            double_way_datasets=double_way_datasets,
                                                                                            loss_func=loss_func)

        logger.info(f'test loss: {test_total_loss:.4f}')

        for metric_name in test_metrics.keys():
            logger.info(
                f'test {metric_name}: {test_metrics[metric_name]:.4f}')

        test_metric_indicator = []
        for metric_name in test_metrics.keys():
            test_metric_indicator.append(
                (metric_name, test_metrics[metric_name], True))
        early_stop = early_stopping.step(test_metric_indicator, model, dataset_name=args.dataset_name)

        if args.dataset_name in ['oag']:
            if test_metrics['acc'] > best_metrics['acc']:
                best_metrics = test_metrics
        else :
            if test_metrics['roc_auc'] > best_metrics['roc_auc']:
                best_metrics = test_metrics
        for metric_name in best_metrics.keys():
            logger.info(
                f'best test {metric_name}: {best_metrics[metric_name]:.4f}')

    sys.exit()