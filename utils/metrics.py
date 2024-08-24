import torch
from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np
from sklearn.metrics import accuracy_score

def get_link_prediction_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the link prediction task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    average_precision = average_precision_score(y_true=labels, y_score=predicts)
    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'average_precision': average_precision, 'roc_auc': roc_auc}


def get_node_classification_metrics_em(predicts: torch.Tensor, labels: torch.Tensor):
    """
    Get metrics for the node classification task.
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        Dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts[:,1].cpu().detach().numpy()
    labels = labels.cpu().numpy()

    unique_labels = np.unique(labels)
    if len(unique_labels) == 1:
        binary_predicts = (predicts >= 0.5).astype(int)  
        accuracy = accuracy_score(y_true=labels, y_pred=binary_predicts)
        return {'accuracy': accuracy}
    else:
        roc_auc = roc_auc_score(y_true=labels, y_score=predicts)
        return {'roc_auc': roc_auc}
    
def get_node_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the node classification task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'roc_auc': roc_auc}
