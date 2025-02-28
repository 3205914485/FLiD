import torch

import os
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn

from utils.metrics import get_link_prediction_metrics
from utils.utils import NegativeEdgeSampler, NeighborSampler

from models.modules import MergeLayer
from utils.utils import convert_to_gpu, create_optimizer
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from evaluate_models_utils import evaluate_model_link_prediction
from utils.metrics import get_link_prediction_metrics
from utils.DataLoader import get_idx_data_loader
from utils.EarlyStopping import EarlyStopping

from models.modules import MergeLayer

from PTCL.trainer import Trainer
from PTCL.M_step import train_model_node_classification_withembeddings


def em_warmup(args, data, logger, Etrainer: Trainer, Mtrainer: Trainer, src_node_embeddings, dst_node_embeddings, pseudo_labels, pseudo_labels_store):

    logger.info("\nStart Warming up\n")
    logger.info("Warm-up-1 : Start training link prediction for backbone\n")
    new_src_embeddings, new_dst_embeddings = link_prediction(
        args=args, Etrainer=Etrainer, data=data, logger=logger)
    src_node_embeddings.copy_(new_src_embeddings)
    dst_node_embeddings.copy_(new_dst_embeddings)
    logger.info("Warm-up-2 : Start training node classification for decoder\n")
    if args.mmodel_name =='mlp':
        save_model_name = f'ncem_{Etrainer.model_name}'
    else:
        save_model_name = f'ncem_{Etrainer.model_name}_{Mtrainer.model_name}'
    save_model_folder = f"./saved_models/ncem/M/warmup/{args.dataset_name}/{args.seed}/{save_model_name}/"
    val_total_loss, val_metrics, test_total_loss, test_metrics =\
        train_model_node_classification_withembeddings(
            args=args,
            data=data,
            logger=logger,
            Etrainer=Etrainer,
            Mtrainer=Mtrainer,
            train=args.warmup_m_train,
            patience=args.mw_patience,
            pseudo_labels=pseudo_labels,
            pseudo_labels_store=pseudo_labels_store,
            save_model_folder=save_model_folder,
            num_epochs=args.num_epochs_m_warmup,
            src_node_embeddings=src_node_embeddings,
            dst_node_embeddings=dst_node_embeddings)

    return val_total_loss, val_metrics, test_total_loss, test_metrics


def link_prediction(args, Etrainer, data, logger):
    node_raw_features = data["node_raw_features"]
    full_data = data["full_data"]
    train_data = data["train_data"]
    val_data = data["val_data"]
    test_data = data["test_data"]
    full_neighbor_sampler = data["full_neighbor_sampler"]
    full_idx_data_loader = data["full_idx_data_loader"]
    train_idx_data_loader = data["train_idx_data_loader"]
    val_idx_data_loader = data["val_idx_data_loader"]
    test_idx_data_loader = data["test_idx_data_loader"]
    train_neighbor_sampler = get_neighbor_sampler(data=train_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                  time_scaling_factor=args.time_scaling_factor, seed=0)

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)

    train_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=train_data.src_node_ids, dst_node_ids=train_data.dst_node_ids)
    val_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=0)
    test_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=2)

    # get data loaders
    train_idx_data_loader = get_idx_data_loader(indices_list=list(
        range(len(train_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    val_idx_data_loader = get_idx_data_loader(indices_list=list(
        range(len(val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(
        range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)

    link_predictor = MergeLayer(input_dim1=node_raw_features.shape[1], input_dim2=node_raw_features.shape[1],
                                hidden_dim=node_raw_features.shape[1], output_dim=1)
    model = nn.Sequential(Etrainer.model, link_predictor)
    model = convert_to_gpu(model, device=args.device)
    model_name = Etrainer.model_name
    # Considering here we do not only train the dyg_backbone , so we create optimizer for the whole model instead of using Trainer.optimizer
    optimizer = create_optimizer(model=model, optimizer_name=args.optimizer,
                                 learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    save_model_name = f'ncem_{model_name}'
    save_model_folder = f"./saved_models/ncem/E/warmup/{args.dataset_name}/{args.seed}/{save_model_name}/"
    # shutil.rmtree(save_model_folder, ignore_errors=True)
    if not os.path.exists(save_model_folder):
        os.makedirs(save_model_folder, exist_ok=True)

    early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
                                   save_model_name=save_model_name, logger=logger, model_name=model_name)

    # Considering the link_prediction loss is not the same as the node_classification task of E/M trainer , so here we directly use BCEloss()
    loss_func = nn.BCELoss()
    if args.warmup_e_train:
        for epoch in range(args.num_epochs_e_warmup):

            model.train()
            if model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']:
                # training, only use training graph
                model[0].set_neighbor_sampler(train_neighbor_sampler)
            if model_name in ['JODIE', 'DyRep', 'TGN']:
                # reinitialize memory of memory-based models at the start of each epoch
                model[0].memory_bank.__init_memory_bank__()

            # store train losses and metrics
            train_losses, train_metrics = [], []
            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                train_data_indices = train_data_indices.numpy()
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                    train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
                    train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices]

                _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(
                    size=len(batch_src_node_ids))
                batch_neg_src_node_ids = batch_src_node_ids

                # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
                # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
                if model_name in ['TGAT', 'CAWN', 'TCL']:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=args.num_neighbors)

                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=args.num_neighbors)
                elif model_name in ['JODIE', 'DyRep', 'TGN']:
                    # note that negative nodes do not change the memories while the positive nodes change the memories,
                    # we need to first compute the embeddings of negative nodes for memory-based models
                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          edge_ids=None,
                                                                          edges_are_positive=False,
                                                                          num_neighbors=args.num_neighbors)

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

                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=args.num_neighbors,
                                                                          time_gap=args.time_gap)
                elif model_name in ['DyGFormer']:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times)

                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times)
                else:
                    raise ValueError(
                        f"Wrong value for model_name {model_name}!")
                # get positive and negative probabilities, shape (batch_size, )
                positive_probabilities = model[1](
                    input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
                negative_probabilities = model[1](
                    input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1).sigmoid()

                predicts = torch.cat(
                    [positive_probabilities, negative_probabilities], dim=0)
                labels = torch.cat([torch.ones_like(
                    positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)

                loss = loss_func(input=predicts, target=labels)

                train_losses.append(loss.item())

                train_metrics.append(get_link_prediction_metrics(
                    predicts=predicts, labels=labels))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_idx_data_loader_tqdm.set_description(
                    f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')

                if model_name in ['JODIE', 'DyRep', 'TGN']:
                    # detach the memories and raw messages of nodes in the memory bank after each batch, so we don't back propagate to the start of time
                    model[0].memory_bank.detach_memory_bank()

            val_losses, val_metrics = evaluate_model_link_prediction(model_name=model_name,
                                                                     model=model,
                                                                     neighbor_sampler=full_neighbor_sampler,
                                                                     evaluate_idx_data_loader=val_idx_data_loader,
                                                                     evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                     evaluate_data=val_data,
                                                                     loss_func=loss_func,
                                                                     num_neighbors=args.num_neighbors,
                                                                     time_gap=args.time_gap)

            logger.info(
                f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.4f}')
            for metric_name in train_metrics[0].keys():
                logger.info(
                    f'train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.4f}')
            logger.info(f'validate loss: {np.mean(val_losses):.4f}')
            for metric_name in val_metrics[0].keys():
                logger.info(
                    f'validate {metric_name}, {np.mean([val_metric[metric_name] for val_metric in val_metrics]):.4f}')

            # perform testing once after test_interval_epochs
            if (epoch + 1) % args.test_interval_epochs == 0:
                test_losses, test_metrics = evaluate_model_link_prediction(model_name=model_name,
                                                                           model=model,
                                                                           neighbor_sampler=full_neighbor_sampler,
                                                                           evaluate_idx_data_loader=test_idx_data_loader,
                                                                           evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                           evaluate_data=test_data,
                                                                           loss_func=loss_func,
                                                                           num_neighbors=args.num_neighbors,
                                                                           time_gap=args.time_gap)

                logger.info(f'test loss: {np.mean(test_losses):.4f}')
                for metric_name in test_metrics[0].keys():
                    logger.info(
                        f'test {metric_name}, {np.mean([test_metric[metric_name] for test_metric in test_metrics]):.4f}')

                test_metric_indicator = []
                for metric_name in test_metrics[0].keys():
                    test_metric_indicator.append((metric_name, np.mean(
                        [test_metric[metric_name] for test_metric in test_metrics]), True))
                early_stop = early_stopping.step(test_metric_indicator, model)

                if early_stop[0]:
                    break

    # load the best model
    early_stopping.load_checkpoint(model,map_location=args.device)
    # generating the embeddings
    # Loop through events and generate embeddings
    # Etrainer.model = model[0]
    model[0].eval()

    logger.info("Loop through all events to generate embeddings\n ")
    src_node_embeddings_list, dst_node_embeddings_list = [], []
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
        src_node_embeddings_list.append(batch_src_node_embeddings)
        dst_node_embeddings_list.append(batch_dst_node_embeddings)

    # Concatenate all embeddings
    src_node_embeddings = torch.cat(src_node_embeddings_list, dim=0).detach()
    dst_node_embeddings = torch.cat(dst_node_embeddings_list, dim=0).detach()

    return src_node_embeddings, dst_node_embeddings
