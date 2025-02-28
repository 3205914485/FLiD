import torch
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
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
    predicts = torch.softmax(predicts, dim=1)
    predicts_np = predicts.cpu().detach().numpy()
    labels_np = labels.cpu().numpy()

    # Get the predicted classes
    predicted_classes = np.argmax(predicts_np, axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true=labels_np, y_pred=predicted_classes)
    
    # Calculate ROC AUC score for each class, then average
    if len(np.unique(labels_np)) > 1:
        if predicts_np.shape[1] == 2:
            roc_auc = roc_auc_score(y_true=labels_np, y_score=predicts_np[:, 1])
            f1 = f1_score(labels_np, predicted_classes, average='binary')
        else:
            roc_auc = roc_auc_score(y_true=labels_np, y_score=predicts_np, multi_class='ovr')
            f1 = f1_score(y_true=labels_np, y_pred=predicted_classes, average='macro')
    else:
        roc_auc = 0.0
        f1 = 0.0

    return {'roc_auc': roc_auc, 'acc': accuracy}
    
def get_node_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the node classification task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = torch.softmax(predicts, dim=1)
    predicts_np = predicts.cpu().detach().numpy()
    labels_np = labels.cpu().numpy()

    # Get the predicted classes
    predicted_classes = np.argmax(predicts_np, axis=1)
    accuracy = accuracy_score(y_true=labels_np, y_pred=predicted_classes)
    # Calculate accuracy
    
    # Calculate ROC AUC score for each class, then average
    if len(np.unique(labels_np)) > 1:
        if predicts_np.shape[1] == 2:
            roc_auc = roc_auc_score(y_true=labels_np, y_score=predicts_np[:, 1])
            f1 = f1_score(labels_np, predicted_classes, average='binary')
        else:
            roc_auc = roc_auc_score(y_true=labels_np, y_score=predicts_np, multi_class='ovr')
            f1 = f1_score(y_true=labels_np, y_pred=predicted_classes, average='macro')
    else:
        roc_auc = 0.0
        f1 = 0.0

    return {'roc_auc': roc_auc, 'acc': accuracy}
