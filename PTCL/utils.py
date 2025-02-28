import json
import os
import numpy as np
import torch
import torch.nn.functional as F

def log_and_save_metrics(logger, phase, loss, metrics, metric_dict, prefix):
    logger.info(f'{phase}: {prefix} loss: {loss:.4f}')
    for metric_name in metrics.keys():
        metric_value = metrics[metric_name]
        logger.info(f'{prefix} {metric_name}, {metric_value:.4f}\n')
        metric_dict[metric_name] = metric_value


def save_results(args, Etrainer, Eval_metric_dict, Etest_metric_dict, Mval_metric_dict, Mtest_metric_dict, run):
    if Etrainer.model_name not in ['JODIE', 'DyRep', 'TGN']:
        result_json = {
            "Evalidate metrics": {metric_name: f'{Eval_metric_dict[metric_name]:.4f}' for metric_name in Eval_metric_dict},
            "Etest metrics": {metric_name: f'{Etest_metric_dict[metric_name]:.4f}' for metric_name in Etest_metric_dict},
            "Mvalidate metrics": {metric_name: f'{Mval_metric_dict[metric_name]:.4f}' for metric_name in Mval_metric_dict},
            "Mtest metrics": {metric_name: f'{Mtest_metric_dict[metric_name]:.4f}' for metric_name in Mtest_metric_dict}
        }
    else:
        result_json = {
            "Etest metrics": {metric_name: f'{Etest_metric_dict[metric_name]:.4f}' for metric_name in Etest_metric_dict},
            "Mtest metrics": {metric_name: f'{Mtest_metric_dict[metric_name]:.4f}' for metric_name in Mtest_metric_dict}
        }
    result_json = json.dumps(result_json, indent=4)

    save_result_folder = f"./saved_results/ncem/{args.prefix}/{run}/{args.dataset_name}"
    os.makedirs(save_result_folder, exist_ok=True)
    save_result_path = os.path.join(
        save_result_folder, f"{args.emodel_name}_{args.mmodel_name}.json")

    with open(save_result_path, 'w') as file:
        file.write(result_json)

def entropy_filter(ps_labels, ps_labels_store, threshold=0.6):

    accumulated_probs = torch.sum(torch.stack(ps_labels_store), dim=0)
    is_double_way = ps_labels.shape[0] == 2
    if is_double_way:
        ps_labels = torch.cat([ps_labels[0],ps_labels[1]], dim=0).reshape([1,-1])
        accumulated_probs = torch.cat([accumulated_probs[0],accumulated_probs[1]],dim=0)

    probs = F.softmax(accumulated_probs, dim=1)  
    log_probs = torch.log2(probs + 1e-10) 
    entropy = -torch.sum(probs * log_probs, dim=1) 

    ps_labels[:,entropy > threshold] = -1 
    if is_double_way: 
        ps_labels = torch.cat([ps_labels[:,:ps_labels.shape[1]//2],ps_labels[:,ps_labels.shape[1]//2:]],dim=0)
    print(f'Entropy Filtering: {sum(entropy<=threshold)/ps_labels.shape[1]:.2f}')
    return ps_labels

def prob_filter(ps_labels, ps_labels_store, threshold=0.6):
    probs = ps_labels_store[-1]
    is_double_way = ps_labels.shape[0] == 2
    if is_double_way:
        ps_labels = torch.cat([ps_labels[0],ps_labels[1]], dim=0).reshape([1,-1])
        probs = torch.cat([probs[0],probs[1]],dim=0)
    probs = torch.max(probs,dim=1)[0]
    ps_labels[:,probs < threshold] = -1 
    if is_double_way: 
        ps_labels = torch.cat([ps_labels[:,:ps_labels.shape[1]//2],ps_labels[:,ps_labels.shape[1]//2:]],dim=0)
    print(f'Probability Filtering: {sum(probs<=threshold)/ps_labels.shape[1]:.2f}')
    return ps_labels   

def update_pseudo_labels(data, pseudo_labels, pseudo_labels_store, double_way_dataset, mode, \
                         use_transductive=0, save=False, save_path=0, threshold=0.6, iter_num=-1, ps_filter='none'):

    if save:
        os.makedirs(save_path, exist_ok=True)  
        torch.save(pseudo_labels, os.path.join(save_path, f'raw_{iter_num}.pt'))  
    else :
        pass

    if ps_filter == 'entropy':
        pseudo_labels = entropy_filter(pseudo_labels, pseudo_labels_store, threshold=threshold)
    elif ps_filter == 'probability':
        pseudo_labels = prob_filter(pseudo_labels, pseudo_labels_store, threshold=threshold)   
    else: 
        pass

    true_labels = data['full_data'].labels
    labels_times = data['full_data'].labels_time
    interact_times = data['full_data'].node_interact_times
    val_offest = data['val_offest']
    train_mask = list(range(pseudo_labels.shape[1])) < val_offest
    if use_transductive:
        if data['dataset_name'] in double_way_dataset:
            mask_gt_u = torch.from_numpy((interact_times == labels_times[0]) & train_mask).to(torch.bool) 
            mask_gt_i = torch.from_numpy((interact_times == labels_times[1]) & train_mask).to(torch.bool)
            pseudo_labels[0,mask_gt_u] = torch.from_numpy(
                true_labels[0][mask_gt_u].astype('float32')).to(pseudo_labels.device)
            pseudo_labels[1,mask_gt_i] = torch.from_numpy(
                true_labels[1][mask_gt_i].astype('float32')).to(pseudo_labels.device)
        else:
            if mode == 'ps': 
                mask_gt = torch.from_numpy((interact_times == labels_times) & train_mask).to(torch.bool)
                pseudo_labels[0,mask_gt] = torch.from_numpy(
                    true_labels[mask_gt].astype('float32')).to(pseudo_labels.device)
            elif mode == 'gt':
                pseudo_labels[0,:] = torch.from_numpy(true_labels.astype('float32')).to(pseudo_labels.device)
    else :
        if data['dataset_name'] in double_way_dataset:
            mask_gt_u = torch.from_numpy(interact_times == labels_times[0]).to(torch.bool) 
            mask_gt_i = torch.from_numpy(interact_times == labels_times[1]).to(torch.bool)
            pseudo_labels[0,mask_gt_u] = torch.from_numpy(
                true_labels[0][mask_gt_u].astype('float32')).to(pseudo_labels.device)
            pseudo_labels[1,mask_gt_i] = torch.from_numpy(
                true_labels[1][mask_gt_i].astype('float32')).to(pseudo_labels.device)
        else:
            if mode == 'ps': 
                mask_gt = torch.from_numpy(interact_times == labels_times).to(torch.bool)
                pseudo_labels[0,mask_gt] = torch.from_numpy(
                    true_labels[mask_gt].astype('float32')).to(pseudo_labels.device)
            elif mode == 'gt':
                pseudo_labels[0,:] = torch.from_numpy(true_labels.astype('float32')).to(pseudo_labels.device)
    if save:
        torch.save(pseudo_labels, os.path.join(save_path, f'updated_{iter_num}.pt'))  

    return pseudo_labels
