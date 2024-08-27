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


def update_pseudo_labels(data, pseudo_labels):

    true_labels = data['full_data'].labels.astype('float32')
    labels_times = data['full_data'].labels_time
    interact_times = data['full_data'].node_interact_times

    mask = torch.from_numpy(interact_times == labels_times).to(torch.bool)

    pseudo_labels[mask] = torch.from_numpy(
        true_labels[mask]).unsqueeze(1).to(pseudo_labels.device)

    return pseudo_labels
