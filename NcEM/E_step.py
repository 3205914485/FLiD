import logging
import time
import sys
import os
from tqdm import tqdm
import numpy as np
import warnings
import shutil
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.TGAT import TGAT
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.CAWN import CAWN
from models.TCL import TCL
from models.GraphMixer import GraphMixer
from models.M import M
from models.DyGFormer import DyGFormer
from models.modules import MergeLayer, MLPClassifier
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from utils.utils import get_neighbor_sampler
from utils.metrics import get_link_prediction_metrics, get_node_classification_metrics_em
from utils.utils import set_random_seed
from utils.utils import NegativeEdgeSampler, NeighborSampler
from utils.DataLoader import Data
from utils.EarlyStopping import EarlyStopping

from NcEM.trainer import Trainer

double_way_dataset = ['bot','bot22','taobao','yelp']

def evaluate_model_node_classification_E_step(model_name: str, model: nn.Module, dataset: str,neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                       evaluate_data: Data, loss_func: nn.Module, num_neighbors: int = 20, time_gap: int = 2000):
    r"""
    evaluate models on the node classification task with E step
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    if model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'M','DyGFormer']:
        # evaluation phase use all the graph information
        model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()

    with torch.no_grad():
        # store evaluate losses, trues and predicts
        evaluate_total_loss, evaluate_y_trues, evaluate_y_predicts, evaluate_y_trues_gt, evaluate_y_predicts_gt= 0.0, [], [], [], []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            if dataset =='bot' or dataset=='bot22':
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_labels = \
                    evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                    evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices], [evaluate_data.labels[0][evaluate_data_indices],evaluate_data.labels[1][evaluate_data_indices]]
            else:
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_labels, batch_labels_times = \
                    evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                    evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices], evaluate_data.labels[evaluate_data_indices],\
                    evaluate_data.labels_time[evaluate_data_indices]

            if model_name in ['TGAT', 'CAWN', 'TCL']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['JODIE', 'DyRep', 'TGN']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      edge_ids=batch_edge_ids,
                                                                      edges_are_positive=True,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['GraphMixer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)
            elif model_name in ['M']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      message_idx=batch_edge_ids,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)                
            elif model_name in ['DyGFormer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times)
            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")
            # get predicted probabilities, shape (batch_size, )
            if dataset in double_way_dataset:
                predicts = model[1](x=torch.cat([batch_src_node_embeddings,batch_dst_node_embeddings],dim=0)).squeeze(dim=-1).sigmoid()
                labels = torch.from_numpy(np.concatenate([batch_labels[0],batch_labels[1]],axis=0)).long().to(predicts.device)
                mask_gt_src = torch.from_numpy(batch_node_interact_times == batch_labels_times[0]).to(torch.bool)
                mask_gt_dst = torch.from_numpy(batch_node_interact_times == batch_labels_times[1]).to(torch.bool)
                mask_gt = torch.cat([mask_gt_src,mask_gt_dst],dim=0).squeeze(dim=-1)
            else :
                predicts = model[1](x=batch_src_node_embeddings).squeeze(dim=-1).sigmoid()
                labels = torch.from_numpy(batch_labels).long().to(predicts.device) 
                mask_gt = torch.from_numpy(batch_node_interact_times == batch_labels_times).to(torch.bool)

            loss = loss_func(input=predicts, target=labels)

            evaluate_total_loss += loss.item()

            predicts_gt, labels_gt = predicts[mask_gt], labels[mask_gt]

            evaluate_y_trues.append(labels)
            evaluate_y_predicts.append(predicts)

            evaluate_y_trues_gt.append(labels_gt)
            evaluate_y_predicts_gt.append(predicts_gt)            

            evaluate_idx_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')

        evaluate_total_loss /= (batch_idx + 1)
        evaluate_y_trues = torch.cat(evaluate_y_trues, dim=0)
        evaluate_y_predicts = torch.cat(evaluate_y_predicts, dim=0)
        evaluate_y_trues_gt = torch.cat(evaluate_y_trues_gt, dim=0)
        evaluate_y_predicts_gt = torch.cat(evaluate_y_predicts_gt, dim=0)        
        evaluate_metrics = get_node_classification_metrics_em(predicts=evaluate_y_predicts, labels=evaluate_y_trues)
        evaluate_metrics_gt = get_node_classification_metrics_em(predicts=evaluate_y_predicts_gt, labels=evaluate_y_trues_gt)

    return evaluate_total_loss, evaluate_metrics, evaluate_metrics_gt

def e_step(Etrainer: Trainer, Mtrainer:Trainer, gt_weight, data, pseudo_labels, args, logger, src_node_embeddings, dst_node_embeddings, pseudo_labels_confidence):
    

    logger.info(f"Starting E-step\n")
    full_data = data['full_data']
    train_data = data["train_data"]
    val_data = data["val_data"]
    test_data = data["test_data"]
    full_neighbor_sampler = data["full_neighbor_sampler"]
    full_idx_data_loader = data["full_idx_data_loader"]
    train_idx_data_loader = data["train_idx_data_loader"]
    val_idx_data_loader = data["val_idx_data_loader"]
    test_idx_data_loader = data["test_idx_data_loader"]

    dynamic_backbone = Etrainer.model
    # node_classifier = Mtrainer.model
    # node_classifier = MLPClassifier(input_dim=172, dropout=args.dropout).to(args.device)
    if args.decoder == 1:
        node_classifier = Mtrainer.model[1]
    else:
        node_classifier = Mtrainer.model[0]
    model = nn.Sequential(dynamic_backbone, node_classifier)
    if args.decoder ==1:
        optimizer = Etrainer.optimizer
    else:
        optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    
    model_name = Etrainer.model_name
    loss_func = Etrainer.criterion
    save_model_name = f'ncem_{model_name}'
    save_model_folder = f"./saved_models/ncem/E/{args.prefix}/{args.dataset_name}/{args.seed}/{save_model_name}/"
    shutil.rmtree(save_model_folder, ignore_errors=True)
    os.makedirs(save_model_folder, exist_ok=True)
    early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
                                        save_model_name=save_model_name, logger=logger, model_name=model_name)
    best_acc, best_acc_gt = 0.0, 0.0
    for epoch in range(args.num_epochs_e_step):
        model.train()
        if model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']:
            # training process, set the neighbor sampler
            model[0].set_neighbor_sampler(full_neighbor_sampler)
        if model_name in ["M"]:
            model[0].set_neighbor_sampler(full_neighbor_sampler)
            #model[0].message_function.resetparameters()
        if model_name in ['JODIE', 'DyRep', 'TGN']:
            # reinitialize memory of memory-based models at the start of each epoch
            model[0].memory_bank.__init_memory_bank__()

        # store train losses, trues and predicts
        train_total_loss, train_y_trues, train_y_predicts = 0.0, [], []
        train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
        for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
            train_data_indices = train_data_indices.numpy()

            if args.dataset_name in double_way_dataset :
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_labels, batch_labels_times, batch_ps_confidence = \
                    train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], train_data.node_interact_times[train_data_indices], \
                    train_data.edge_ids[train_data_indices], [pseudo_labels[0][train_data_indices],pseudo_labels[1][train_data_indices]], \
                    [train_data.labels_time[0][train_data_indices], train_data.labels_time[1][train_data_indices]], pseudo_labels_confidence[train_data_indices]
            else :
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_labels, batch_labels_times, batch_ps_confidence = \
                train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], train_data.node_interact_times[train_data_indices], \
                train_data.edge_ids[train_data_indices], pseudo_labels[train_data_indices], train_data.labels_time[train_data_indices], pseudo_labels_confidence[train_data_indices]

            if model_name in ['TGAT', 'CAWN', 'TCL']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
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
            elif model_name in ['M']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                        dst_node_ids=batch_dst_node_ids,
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
            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")
        # get predicted probabilities, shape (batch_size, )
            if args.dataset_name in double_way_dataset:
                predicts = model[1](x=torch.cat([batch_src_node_embeddings,batch_dst_node_embeddings],dim=0))
                labels = torch.cat([batch_labels[0],batch_labels[1]],axis=0).to(torch.long).to(predicts.device).squeeze(dim=-1)
                mask_gt_src = torch.from_numpy(batch_node_interact_times == batch_labels_times[0]).to(torch.bool)
                mask_gt_dst = torch.from_numpy(batch_node_interact_times == batch_labels_times[1]).to(torch.bool)
                mask_gt = torch.cat([mask_gt_src,mask_gt_dst],dim=0).squeeze(dim=-1)

            else :
                predicts = model[1](x=batch_src_node_embeddings)
                labels = batch_labels.to(torch.long).to(predicts.device).squeeze(dim=-1) 
                mask_gt = torch.from_numpy(batch_node_interact_times == batch_labels_times).to(torch.bool)
            if args.use_confidence:
                mask_ps = (batch_ps_confidence > args.confidence_threshold).to(torch.bool).squeeze(dim=-1) 
                mask_ps &= ~mask_gt
            else:
                mask_ps = ~mask_gt
            predicts_gt, labels_gt = predicts[mask_gt], labels[mask_gt]
            predicts_ps, labels_ps = predicts[mask_ps], labels[mask_ps] 
            loss_gt = loss_func(input = predicts_gt, target = labels_gt)
            loss_ps = loss_func(input = predicts_ps, target = labels_ps)
            loss = gt_weight*loss_gt + (1-gt_weight)*loss_ps
            train_total_loss += loss.item()
            train_y_trues.append(labels)
            train_y_predicts.append(predicts)      

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            train_idx_data_loader_tqdm.set_description(f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')

            if model_name in ['JODIE', 'DyRep', 'TGN']:
                # detach the memories and raw messages of nodes in the memory bank after each batch, so we don't back propagate to the start of time
                model[0].memory_bank.detach_memory_bank()

        train_total_loss /= (batch_idx + 1)
        train_y_trues = torch.cat(train_y_trues, dim=0)
        train_y_predicts = torch.cat(train_y_predicts, dim=0)

        train_metrics = get_node_classification_metrics_em(predicts=train_y_predicts, labels=train_y_trues)

        val_total_loss, val_metrics, val_metrics_gt = evaluate_model_node_classification_E_step(model_name=model_name,
                                                                            model=model,
                                                                            dataset=args.dataset_name,
                                                                            neighbor_sampler=full_neighbor_sampler,
                                                                            evaluate_idx_data_loader=val_idx_data_loader,
                                                                            evaluate_data=val_data,
                                                                            loss_func=loss_func,
                                                                            num_neighbors=args.num_neighbors,
                                                                            time_gap=args.time_gap)

        logger.info(f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {train_total_loss:.4f}')
        for metric_name in train_metrics.keys():
            logger.info(f'train {metric_name}, {train_metrics[metric_name]:.4f}')
        logger.info(f'validate loss: {val_total_loss:.4f}')
        for metric_name in val_metrics.keys():
            logger.info(f'validate {metric_name}, {val_metrics[metric_name]:.4f}')
        for metric_name in val_metrics_gt.keys():
            logger.info(f'Ground Truth validate {metric_name}, {val_metrics_gt[metric_name]:.4f}')
        # perform testing once after test_interval_epochs
        if (epoch + 1) % args.test_interval_epochs == 0:
            if model_name in ['JODIE', 'DyRep', 'TGN']:
                # backup memory bank after validating so it can be used for testing nodes (since test edges are strictly later in time than validation edges)
                val_backup_memory_bank = model[0].memory_bank.backup_memory_bank()

            test_total_loss, test_metrics, test_metrics_gt = evaluate_model_node_classification_E_step(model_name=model_name,
                                                                                model=model,
                                                                                dataset=args.dataset_name,
                                                                                neighbor_sampler=full_neighbor_sampler,
                                                                                evaluate_idx_data_loader=test_idx_data_loader,
                                                                                evaluate_data=test_data,
                                                                                loss_func=loss_func,
                                                                                num_neighbors=args.num_neighbors,
                                                                                time_gap=args.time_gap)

            if model_name in ['JODIE', 'DyRep', 'TGN']:
                # reload validation memory bank for saving models
                # note that since model treats memory as parameters, we need to reload the memory to val_backup_memory_bank for saving models
                model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)

            logger.info(f'test loss: {test_total_loss:.4f}')
            for metric_name in test_metrics.keys():
                logger.info(f'test {metric_name}, {test_metrics[metric_name]:.4f}')
            for metric_name in test_metrics_gt.keys():
                logger.info(f'Groun Truth test {metric_name}, {test_metrics_gt[metric_name]:.4f}')
        # select the best model based on all the validate metrics
        test_metric_indicator = []
        for metric_name in test_metrics.keys():
            test_metric_indicator.append((metric_name, test_metrics[metric_name], True))
        early_stop = early_stopping.step(test_metric_indicator, model)

        if test_metrics['roc_auc']>best_acc:
            best_acc = test_metrics['roc_auc']
        logger.info(f'Best test {metric_name}: {best_acc:.4f}')
        if test_metrics_gt['roc_auc']>best_acc_gt:
            best_acc_gt = test_metrics_gt['roc_auc']
        logger.info(f'Best Ground Truth test {metric_name}: {best_acc_gt:.4f}')

        if early_stop[0]:
            break
    # load the best model
    early_stopping.load_checkpoint(model)

    # evaluate the best model
    logger.info(f'get best performance on dataset {args.dataset_name}...')

    # the saved best model of memory-based models cannot perform validation since the stored memory has been updated by validation data
    if model_name not in ['JODIE', 'DyRep', 'TGN']:
        val_total_loss, val_metric, val_metrics_gt = evaluate_model_node_classification_E_step(model_name=model_name,
                                                                            model=model,
                                                                            dataset=args.dataset_name,
                                                                            neighbor_sampler=full_neighbor_sampler,
                                                                            evaluate_idx_data_loader=val_idx_data_loader,
                                                                            evaluate_data=val_data,
                                                                            loss_func=loss_func,
                                                                            num_neighbors=args.num_neighbors,
                                                                            time_gap=args.time_gap)

    test_total_loss, test_metrics, test_metrics_gt = evaluate_model_node_classification_E_step(model_name=model_name,
                                                                        model=model,
                                                                        dataset=args.dataset_name,
                                                                        neighbor_sampler=full_neighbor_sampler,
                                                                        evaluate_idx_data_loader=test_idx_data_loader,
                                                                        evaluate_data=test_data,
                                                                        loss_func=loss_func,
                                                                        num_neighbors=args.num_neighbors,
                                                                        time_gap=args.time_gap)
    for metric_name in test_metrics_gt.keys():
        logger.info(f'test {metric_name}, {test_metrics_gt[metric_name]:.4f}')
    # generating the embeddings
        # Loop through events and generate embeddings
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
    new_src_embeddings = torch.cat(src_node_embeddings_list, dim=0)
    new_dst_embeddings = torch.cat(dst_node_embeddings_list, dim=0)
    src_node_embeddings.copy_(new_src_embeddings)
    dst_node_embeddings.copy_(new_dst_embeddings)
    return val_total_loss, val_metrics, test_total_loss, test_metrics