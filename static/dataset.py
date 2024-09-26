import torch
import numpy as np
import pandas as pd

def load_dataset(dataset_name, val_ratio, test_ratio, device):
    graph = pd.read_csv(f'./processed_data/{dataset_name}/ml_{dataset_name}.csv')
    x = np.load(f'./processed_data/{dataset_name}/static_{dataset_name}.npy')
    edge_type = np.load(f'./processed_data/{dataset_name}/static_{dataset_name}_edge_type.npy')
    labels = pd.read_csv(f'./processed_data/{dataset_name}/static_{dataset_name}_labels.csv')
    edge_index = np.load(f'./processed_data/{dataset_name}/static_{dataset_name}_edge_index.npy') 
    r_labels = labels['0'].values
    src_node_ids = graph.u.values.astype(np.longlong)
    dst_node_ids = graph.i.values.astype(np.longlong)
    node_interact_times = graph.ts.values.astype(np.float64)

    double_way_datasets = ['bot','bot22','dgraph','dsub']

    if dataset_name in double_way_datasets :
        label1 = graph.label_u.values
        label2 = graph.label_i.values
        labels_time1 = graph.last_u_ts.values
        labels_time2 = graph.last_i_ts.values
        labels=[label1,label2]
        labels_time = [labels_time1,labels_time2]
    else:
        labels=graph.label.values
        labels_time = graph.last_timestamp.values

    # spilt based on the gt
    if dataset_name in double_way_datasets:

        merged_node_interact_times = np.zeros(shape=(len(node_interact_times)*2,))
        merged_node_interact_times[0::2] = node_interact_times
        merged_node_interact_times[1::2] = node_interact_times
        merged_labels_time = np.zeros(shape=(len(labels_time[0])*2,))
        merged_labels_time[0::2] = labels_time[0]
        merged_labels_time[1::2] = labels_time[1]
        merged_labels = np.zeros(shape=(len(labels[0])*2,))
        merged_labels[0::2] = labels[0]
        merged_labels[1::2] = labels[1]
        merged_ids = np.zeros(shape=(len(src_node_ids)*2,))
        merged_ids[0::2] = src_node_ids
        merged_ids[1::2] = dst_node_ids

        if dataset_name in ['dsub', 'dgraph']:
            labels_mask = np.isin(merged_labels, [0, 1])
            times_mask = merged_node_interact_times == merged_labels_time
            mask = labels_mask & times_mask
            ground_truth_times = merged_node_interact_times[mask]
        else:    
            mask = merged_node_interact_times == merged_labels_time
            ground_truth_times = merged_node_interact_times[mask]
            
        val_time, test_time = list(np.quantile(ground_truth_times, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))
        train_nodes_mask = merged_node_interact_times <= val_time
        train_nodes = merged_ids[train_nodes_mask & mask].astype(int)
        val_nodes_mask = np.logical_and(merged_node_interact_times <= test_time, merged_node_interact_times > val_time)
        val_nodes = merged_ids[val_nodes_mask & mask].astype(int)
        test_nodes_mask = merged_node_interact_times > test_time
        test_nodes = merged_ids[test_nodes_mask & mask].astype(int)            

    else :
        mask = node_interact_times == labels_time
        ground_truth_times = node_interact_times[mask]
        val_time, test_time = list(np.quantile(ground_truth_times, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))
        train_nodes_mask = node_interact_times <= val_time
        train_nodes = src_node_ids[train_nodes_mask & mask].astype(int)
        val_nodes_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
        val_nodes = src_node_ids[val_nodes_mask & mask].astype(int)
        test_nodes_mask = node_interact_times > test_time
        test_nodes = src_node_ids[test_nodes_mask & mask].astype(int)       

    train_nodes = np.unique(train_nodes) 
    val_nodes = np.unique(val_nodes) 
    test_nodes = np.unique(test_nodes) 

    
    return torch.tensor(x,dtype=torch.float32).to(device), torch.tensor(r_labels,dtype=torch.long).to(device), torch.tensor(edge_index).to(device), torch.tensor(edge_type).to(device), torch.tensor(train_nodes).to(device), torch.tensor(val_nodes).to(device), torch.tensor(test_nodes).to(device)