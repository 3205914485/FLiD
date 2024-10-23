import json
import os
import numpy as np
import torch


def log_and_save_metrics(logger, phase, loss, metrics, metric_dict, prefix):
    logger.info(f'{phase}: {prefix} loss: {loss:.4f}')
    for metric_name in metrics.keys():
        metric_value = metrics[metric_name]
        logger.info(f'{prefix} {metric_name}, {metric_value:.4f}\n')
        metric_dict[metric_name] = metric_value


def save_results(args, Etrainer, Eval_metric_dict, Etest_metric_dict, Mval_metric_dict, Mtest_metric_dict):
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

    save_result_folder = f"./saved_results/ncem/{args.prefix}/{args.dataset_name}"
    os.makedirs(save_result_folder, exist_ok=True)
    save_result_path = os.path.join(
        save_result_folder, f"{args.emodel_name}_{args.mmodel_name}.json")

    with open(save_result_path, 'w') as file:
        file.write(result_json)


def log_average_metrics(logger, metric_all_runs, prefix):
    for metric_name in metric_all_runs[0].keys():
        metric_values = [single_run[metric_name]
                         for single_run in metric_all_runs]
        logger.info(f'{prefix} {metric_name}, {metric_values}')
        logger.info(f'average {prefix} {metric_name}, {np.mean(metric_values):.4f} '
                    f'± {np.std(metric_values, ddof=1):.4f}')


def update_pseudo_labels(data, pseudo_labels, pseudo_entropy, threshold, use_pseudo_entropy, double_way_dataset, use_transductive=0, save=False, save_path=0, iter_num=0):

    if save:
        os.makedirs(save_path, exist_ok=True)  
        torch.save(pseudo_labels, os.path.join(save_path, f'raw_{iter_num}.pt'))  

    pseudo_entropy_list = list(pseudo_entropy)
    num_targets = len(pseudo_entropy_list)

    if use_pseudo_entropy:
        pseudo_entropy_list = [pseudo_confidence - 0.5 for pseudo_confidence in pseudo_entropy_list]
        pseudo_entropy_score = torch.abs(torch.sum(torch.stack(pseudo_entropy_list),dim=0))
        mask_entropy = pseudo_entropy_score > (threshold - 0.5) * num_targets
        pseudo_labels[~mask_entropy] = -1
    else :
        pass
    true_labels = data['full_data'].labels
    labels_times = data['full_data'].labels_time
    interact_times = data['full_data'].node_interact_times
    val_offest = data['val_offest']
    train_mask = list(range(pseudo_labels.shape[1])) < val_offest
    if not use_transductive:
        if data['dataset_name'] in double_way_dataset:
            mask_gt_u = torch.from_numpy(interact_times == labels_times[0]).to(torch.bool) 
            mask_gt_i = torch.from_numpy(interact_times == labels_times[1]).to(torch.bool)
            pseudo_labels[0,mask_gt_u] = torch.from_numpy(
                true_labels[0][mask_gt_u].astype('float32')).to(pseudo_labels.device)
            pseudo_labels[1,mask_gt_i] = torch.from_numpy(
                true_labels[1][mask_gt_i].astype('float32')).to(pseudo_labels.device)
        else:
            mask_gt = torch.from_numpy(interact_times == labels_times).to(torch.bool)
            pseudo_labels[mask_gt] = torch.from_numpy(
                true_labels[mask_gt].astype('float32')).unsqueeze(1).to(pseudo_labels.device)
    else :
        if data['dataset_name'] in double_way_dataset:
            mask_gt_u = torch.from_numpy((interact_times == labels_times[0]) & train_mask).to(torch.bool) 
            mask_gt_i = torch.from_numpy((interact_times == labels_times[1]) & train_mask).to(torch.bool)
            pseudo_labels[0,mask_gt_u] = torch.from_numpy(
                true_labels[0][mask_gt_u].astype('float32')).to(pseudo_labels.device)
            pseudo_labels[1,mask_gt_i] = torch.from_numpy(
                true_labels[1][mask_gt_i].astype('float32')).to(pseudo_labels.device)
        else:
            mask_gt = torch.from_numpy((interact_times == labels_times) and train_mask).to(torch.bool)
            pseudo_labels[mask_gt] = torch.from_numpy(
                true_labels[mask_gt].astype('float32')).unsqueeze(1).to(pseudo_labels.device)

    if save:
        torch.save(pseudo_labels, os.path.join(save_path, f'updated_{iter_num}.pt'))  

    return pseudo_labels, num_targets
