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
from utils.load_configs import get_node_classification_em_args
from tqdm import tqdm

from NcEM.EM_init import em_init
from NcEM.utils import log_and_save_metrics, log_average_metrics, save_results, update_pseudo_labels

cpu_num = 2
os.environ["OMP_NUM_THREADS"] = str(cpu_num)  # noqa
os.environ["MKL_NUM_THREADS"] = str(cpu_num)  # noqa
torch.set_num_threads(cpu_num)

if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    double_way_datasets = ['bot','bot22','dgraph','dsub','yelp','arxiv','oag']
    # get arguments
    args = get_node_classification_em_args()
    # get data for training, validation and testing
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, num_interactions, \
        num_node_features, val_offest, test_offest, train_nodes, test_nodes, num_classes, ps_batch_mask = \
        get_NcEM_data(dataset_name=args.dataset_name, val_ratio=args.val_ratio,
                      test_ratio=args.test_ratio, new_spilt=args.new_spilt, em_patience=args.em_patience)
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
        pseudo_labels_save_path = f"processed_data/{args.dataset_name}/pseudo_labels/{args.emodel_name}/{args.seed}/"

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

        Etrainer, Mtrainer = em_init(args=args,
                                     logger=logger,
                                     train_data=train_data,
                                     node_raw_features=node_raw_features,
                                     edge_raw_features=edge_raw_features,
                                     full_neighbor_sampler=full_neighbor_sampler
                                     )

        model_name = Etrainer.model_name
        save_model_name = f'ncem_{model_name}'
        save_model_folder = f"./saved_models/ncem/EM/{args.prefix}/{args.dataset_name}/{args.seed}/{save_model_name}/"
        early_stopping = EarlyStopping(patience=args.em_patience, save_model_folder=save_model_folder,
                                       save_model_name=save_model_name, logger=logger, model_name=model_name)
        model = nn.Sequential(Etrainer.model,Mtrainer.model)
        early_stopping.load_checkpoint(model,map_location=args.device)

        model[0].eval()

        logger.info("Loop through all events to foward pseudo labels\n ")
        src_node_embeddings_list, dst_node_embeddings_list, pseudo_labels_list = [], [], []
        if model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']:
            model[0].set_neighbor_sampler(full_neighbor_sampler)
        if model_name in ['JODIE', 'DyRep', 'TGN']:
            model[0].memory_bank.__init_memory_bank__()

        idx_data_loader_tqdm = tqdm(full_idx_data_loader, ncols=120)
        for batch_idx, data_indices in enumerate(idx_data_loader_tqdm):
            data_indices = data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                full_data.src_node_ids[data_indices], full_data.dst_node_ids[data_indices], full_data.node_interact_times[data_indices], \
                full_data.edge_ids[data_indices]

            with torch.no_grad():
                if model_name in ['TGAT', 'CAWN', 'TCL']:
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                        dst_node_ids=batch_dst_node_ids,
                                                                        node_interact_times=batch_node_interact_times,
                                                                        num_neighbors=args.num_neighbors)
                elif model_name in ['JODIE', 'DyRep', 'TGN']:
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

                if args.dataset_name in double_way_datasets:
                    predicts = model[1][1](x=torch.cat(
                        [batch_src_node_embeddings, batch_dst_node_embeddings], dim=0))

                else:
                    predicts = model[1][1](x=batch_src_node_embeddings)
            probabilities = torch.softmax(predicts, dim=1)
            _, one_hot_predicts = torch.max(probabilities, dim=1)

            if args.dataset_name in double_way_datasets:
                one_hot_predicts = torch.stack([one_hot_predicts[:one_hot_predicts.shape[0]//2],one_hot_predicts[one_hot_predicts.shape[0]//2:]],dim=0)
                pseudo_labels_list.append(one_hot_predicts.to(torch.long))
            else:
                pseudo_labels_list.append(one_hot_predicts.to(torch.long))


        if args.dataset_name in double_way_datasets: 
            new_labels = torch.cat(pseudo_labels_list, dim=1).detach()
        else:
            new_labels = torch.cat(pseudo_labels_list, dim=0).detach().unsqueeze(dim=0)
        pseudo_labels.copy_(new_labels)
        pseudo_labels = update_pseudo_labels(
            data=data, pseudo_labels=pseudo_labels, save_path=pseudo_labels_save_path, pseudo_labels_store=[],mode='ps',\
            double_way_dataset=double_way_datasets, use_transductive=args.use_transductive,iter_num=0,save=args.save_pseudo_labels)

    sys.exit()
