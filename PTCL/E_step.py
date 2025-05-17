import torch
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

from utils.metrics import get_link_prediction_metrics, get_node_classification_metrics_em
from utils.utils import NegativeEdgeSampler, NeighborSampler
from utils.DataLoader import Data
from models.TGAT import TGAT
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.TCL import TCL
from models.GraphMixer import GraphMixer
from models.DyGFormer import DyGFormer
from models.modules import MergeLayer, MLPClassifier
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from evaluate_models_utils import evaluate_model_link_prediction
from utils.metrics import get_link_prediction_metrics
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args
from models.modules import MergeLayer

from PTCL.trainer import Trainer

def evaluate_model_node_classification_withembeddings(model: nn.Module, dataset: str, src_node_embeddings: torch.tensor,
                                                      dst_node_embeddings: torch.tensor, evaluate_idx_data_loader: DataLoader,
                                                      evaluate_data: Data, loss_func: nn.Module, double_way_datasets: list):

    model.eval()

    with torch.no_grad():
        # store evaluate losses, trues and predicts
        evaluate_total_loss, evaluate_y_trues, evaluate_y_predicts, batch_count = 0.0, [], [], 0.0
        evaluate_idx_data_loader_tqdm = tqdm(
            evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            if dataset in double_way_datasets:
                batch_node_interact_times, batch_edge_ids, batch_labels, batch_labels_times = \
                    evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices], \
                    [evaluate_data.labels[0][evaluate_data_indices],evaluate_data.labels[1][evaluate_data_indices]], \
                    [evaluate_data.labels_time[0][evaluate_data_indices],evaluate_data.labels_time[1][evaluate_data_indices]]    
            else:
                batch_node_interact_times, batch_edge_ids, batch_labels, batch_labels_times = \
                    evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices], \
                    evaluate_data.labels[evaluate_data_indices], evaluate_data.labels_time[evaluate_data_indices]

            batch_src_node_embeddings, batch_dst_node_embeddings = \
                src_node_embeddings[batch_edge_ids-1], dst_node_embeddings[batch_edge_ids-1]

            # get predicted probabilities, shape (batch_size, )
            if dataset in double_way_datasets:
                predicts = model(x=torch.cat(
                    [batch_src_node_embeddings, batch_dst_node_embeddings], dim=0))
                labels = torch.from_numpy(np.concatenate(
                    [batch_labels[0], batch_labels[1]], axis=0)).long().to(predicts.device)
                if dataset == 'dsub':
                    mask_gt_src = torch.from_numpy(
                        (batch_node_interact_times == batch_labels_times[0]) & (np.isin(batch_labels[0],[0,1]))).to(torch.bool)
                    mask_gt_dst = torch.from_numpy(
                        (batch_node_interact_times == batch_labels_times[1]) & (np.isin(batch_labels[1],[0,1]))).to(torch.bool)
                else :
                    mask_gt_src = torch.from_numpy(
                        (batch_node_interact_times == batch_labels_times[0])).to(torch.bool)
                    mask_gt_dst = torch.from_numpy(
                        (batch_node_interact_times == batch_labels_times[1])).to(torch.bool)
                mask = torch.cat([mask_gt_src, mask_gt_dst],dim=0).squeeze(dim=-1)
                probabilities = torch.softmax(predicts, dim=1)
            else:
                predicts = model(x=batch_src_node_embeddings)
                labels = torch.from_numpy(
                    batch_labels).long().to(predicts.device)
                mask = torch.from_numpy(
                    batch_node_interact_times == batch_labels_times).to(torch.bool)
            filtered_predicts = predicts[mask]
            filtered_labels = labels[mask]

            if filtered_predicts.size(0) > 0:
                loss = loss_func(input=filtered_predicts,
                                 target=filtered_labels).mean()
                loss_value = loss.item()
                batch_count += 1
            else:
                loss = torch.tensor(0.0)
                loss_value = 0.0

            evaluate_total_loss += loss_value
            evaluate_y_trues.append(filtered_labels)
            evaluate_y_predicts.append(filtered_predicts)

            evaluate_idx_data_loader_tqdm.set_description(
                f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss_value}')

        evaluate_total_loss /= (batch_count)
        evaluate_y_trues = torch.cat(evaluate_y_trues, dim=0)
        evaluate_y_predicts = torch.cat(evaluate_y_predicts, dim=0)
        evaluate_metrics = get_node_classification_metrics_em(
            predicts=evaluate_y_predicts, labels=evaluate_y_trues)

    return evaluate_total_loss, evaluate_metrics


def train_model_node_classification_withembeddings(args, Mtrainer, Etrainer, data, logger, save_model_folder, patience, train,
                                                   src_node_embeddings, dst_node_embeddings, pseudo_labels, pseudo_labels_store, num_epochs):

    double_way_datasets = args.double_way_datasets
    full_data = data['full_data']
    train_data = data["train_data"]
    val_data = data["val_data"]
    test_data = data["test_data"]
    full_idx_data_loader = data["full_idx_data_loader"]
    train_idx_data_loader = data["train_idx_data_loader"]
    val_idx_data_loader = data["val_idx_data_loader"]
    test_idx_data_loader = data["test_idx_data_loader"]

    model = Etrainer.model[1]
    optimizer = Etrainer.optimizer
    model_name = Etrainer.model_name
    save_model_name = f'ptcl_{model_name}'
    # shutil.rmtree(save_model_folder, ignore_errors=True)
    if not os.path.exists(save_model_folder):
        os.makedirs(save_model_folder, exist_ok=True)
    early_stopping = EarlyStopping(patience=patience, save_model_folder=save_model_folder,
                                   save_model_name=save_model_name, logger=logger, model_name=model_name)
    loss_func = Etrainer.criterion
    test_total_loss, test_metrics = evaluate_model_node_classification_withembeddings(model=model,
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
    if train:
        train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
        best_metrics, best_epoch = {'roc_auc': 0.0, 'acc': 0.0}, 0
        for epoch in range(num_epochs):

            train_total_loss, train_y_trues, train_y_predicts, batch_count = 0.0, [], [], 0.0
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
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
                    predicts = model(x=torch.cat(
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
                    predicts = model(x=batch_src_node_embeddings)
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

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    batch_count += 1
                    train_idx_data_loader_tqdm.set_description(
                        f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')
                else:
                    #  0 loss for this batch
                    train_idx_data_loader_tqdm.set_description(
                        f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, no valid predictions')
                    
            train_total_loss /= (batch_count)
            train_y_trues = torch.cat(train_y_trues, dim=0)
            train_y_predicts = torch.cat(train_y_predicts, dim=0)

            train_metrics = get_node_classification_metrics_em(
                predicts=train_y_predicts, labels=train_y_trues)

            val_total_loss, val_metrics = evaluate_model_node_classification_withembeddings(model=model,
                                                                                            dataset=args.dataset_name,
                                                                                            src_node_embeddings=src_node_embeddings,
                                                                                            dst_node_embeddings=dst_node_embeddings,
                                                                                            evaluate_idx_data_loader=val_idx_data_loader,
                                                                                            evaluate_data=val_data,
                                                                                            double_way_datasets=double_way_datasets,
                                                                                            loss_func=loss_func)

            logger.info(
                f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {train_total_loss:.4f}')
            for metric_name in train_metrics.keys():
                logger.info(
                    f'train {metric_name}, {train_metrics[metric_name]:.4f}')
            logger.info(f'validate loss: {val_total_loss:.4f}')
            for metric_name in val_metrics.keys():
                logger.info(
                    f'validate {metric_name}, {val_metrics[metric_name]:.4f}')

            # perform testing once after test_interval_epochs
            if (epoch + 1) % args.test_interval_epochs == 0:
                test_total_loss, test_metrics = evaluate_model_node_classification_withembeddings(model=model,
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
                    best_epoch = epoch
            else :
                if test_metrics['roc_auc'] > best_metrics['roc_auc']:
                    best_metrics = test_metrics
                    best_epoch = epoch
            for metric_name in best_metrics.keys():
                logger.info(
                    f'best test {metric_name}: {best_metrics[metric_name]:.4f}')

            if early_stop[0]:
                break

    # load the best model
    early_stopping.load_checkpoint(model,map_location=args.device)
    # evaluate the best model
    logger.info(f'get best performance on dataset {args.dataset_name}...')
    val_total_loss, val_metrics = evaluate_model_node_classification_withembeddings(model=model,
                                                                                    dataset=args.dataset_name,
                                                                                    src_node_embeddings=src_node_embeddings,
                                                                                    dst_node_embeddings=dst_node_embeddings,
                                                                                    evaluate_idx_data_loader=val_idx_data_loader,
                                                                                    evaluate_data=val_data,
                                                                                    double_way_datasets=double_way_datasets,
                                                                                    loss_func=loss_func)
    test_total_loss, test_metrics = evaluate_model_node_classification_withembeddings(model=model,
                                                                                      dataset=args.dataset_name,
                                                                                      src_node_embeddings=src_node_embeddings,
                                                                                      dst_node_embeddings=dst_node_embeddings,
                                                                                      evaluate_idx_data_loader=test_idx_data_loader,
                                                                                      evaluate_data=test_data,
                                                                                      double_way_datasets=double_way_datasets,
                                                                                      loss_func=loss_func)

    # generating the pseudo labels for all time nodes
    model.eval()
    pseudo_labels_list, probabilities_list = [], []
    logger.info("Loop through all events to generate pseudo labels\n ")
    full_idx_data_loader_tqdm = tqdm(full_idx_data_loader, ncols=120)
    for batch_idx, full_data_indices in enumerate(full_idx_data_loader_tqdm):
        full_data_indices = full_data_indices.numpy()

        with torch.no_grad():
            if args.dataset_name in double_way_datasets:
                batch_node_interact_times, batch_edge_ids, batch_labels = \
                    full_data.node_interact_times[full_data_indices], full_data.edge_ids[full_data_indices], \
                    [full_data.labels[0][full_data_indices],
                        full_data.labels[1][full_data_indices]]
            else:
                batch_node_interact_times, batch_edge_ids, batch_labels = \
                    full_data.node_interact_times[full_data_indices], full_data.edge_ids[full_data_indices], \
                    full_data.labels[full_data_indices]

            batch_src_node_embeddings, batch_dst_node_embeddings = \
                src_node_embeddings[batch_edge_ids -
                                    1], dst_node_embeddings[batch_edge_ids-1]

            if args.dataset_name in double_way_datasets:
                predicts = model(x=torch.cat(
                    [batch_src_node_embeddings, batch_dst_node_embeddings], dim=0))

            else:
                predicts = model(x=batch_src_node_embeddings)
        probabilities = torch.softmax(predicts, dim=1)
        _, one_hot_predicts = torch.max(probabilities, dim=1)

        if args.dataset_name in double_way_datasets:
            one_hot_predicts = torch.stack([one_hot_predicts[:one_hot_predicts.shape[0]//2],one_hot_predicts[one_hot_predicts.shape[0]//2:]],dim=0)
            pseudo_labels_list.append(one_hot_predicts.to(torch.long))
            probabilities = torch.stack([probabilities[:probabilities.shape[0]//2,:],probabilities[probabilities.shape[0]//2:,:]], dim=0)
            probabilities_list.append(probabilities)
        else:
            pseudo_labels_list.append(one_hot_predicts.to(torch.long))
            probabilities_list.append(probabilities)

    if args.dataset_name in double_way_datasets: 
        new_labels = torch.cat(pseudo_labels_list, dim=1).detach()
    else:
        new_labels = torch.cat(pseudo_labels_list, dim=0).detach()
        probabilities_list = torch.cat(probabilities_list, dim=0).detach()
    pseudo_labels.copy_(new_labels)
    pseudo_labels_store.append(probabilities_list)
    return val_total_loss, val_metrics, test_total_loss, test_metrics


def e_step(Mtrainer, Etrainer, data, args, logger, src_node_embeddings, dst_node_embeddings, pseudo_labels, pseudo_labels_store):
    logger.info("Starting e-step \n")
    save_model_name = f'ptcl_{Mtrainer.model_name}'
    save_model_folder = f"./saved_models/ptcl/e/{args.prefix}/{args.dataset_name}/{args.seed}/{save_model_name}/"
    val_total_loss, val_metrics, test_total_loss, test_metrics = \
        train_model_node_classification_withembeddings(args=args,
                                                       data=data,
                                                       logger=logger,
                                                       Mtrainer=Mtrainer,
                                                       Etrainer=Etrainer,
                                                       train=True,
                                                       patience=args.patience,
                                                       pseudo_labels=pseudo_labels,
                                                       pseudo_labels_store=pseudo_labels_store,
                                                       save_model_folder=save_model_folder,
                                                       num_epochs=args.num_epochs_m_step,
                                                       src_node_embeddings=src_node_embeddings,
                                                       dst_node_embeddings=dst_node_embeddings)
    return val_total_loss, val_metrics, test_total_loss, test_metrics
