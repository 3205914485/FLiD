from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import pandas as pd


class CustomizedDataset(Dataset):
    def __init__(self, indices_list: list):
        """
        Customized dataset.
        :param indices_list: list, list of indices
        """
        super(CustomizedDataset, self).__init__()

        self.indices_list = indices_list

    def __getitem__(self, idx: int):
        """
        get item at the index in self.indices_list
        :param idx: int, the index
        :return:
        """
        return self.indices_list[idx]

    def __len__(self):
        return len(self.indices_list)


def get_idx_data_loader(indices_list: list, batch_size: int, shuffle: bool):
    """
    get data loader that iterates over indices
    :param indices_list: list, list of indices
    :param batch_size: int, batch size
    :param shuffle: boolean, whether to shuffle the data
    :return: data_loader, DataLoader
    """
    dataset = CustomizedDataset(indices_list=indices_list)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             drop_last=False)
    return data_loader


class Data:

    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, edge_ids: np.ndarray, labels: np.ndarray, labels_time: np.ndarray=None):
        """
        Data object to store the nodes interaction information.
        :param src_node_ids: ndarray
        :param dst_node_ids: ndarray
        :param node_interact_times: ndarray
        :param edge_ids: ndarray
        :param labels: ndarray
        """
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.node_interact_times = node_interact_times
        self.edge_ids = edge_ids
        self.labels = labels
        self.labels_time = labels_time
        self.num_interactions = len(src_node_ids)
        self.unique_node_ids = set(src_node_ids) | set(dst_node_ids)
        self.num_unique_nodes = len(self.unique_node_ids)

def check_node_in_dataset(node_id, src_ids, dst_ids):
    return np.any(src_ids == node_id) or np.any(dst_ids == node_id)

def get_link_prediction_data(dataset_name: str, val_ratio: float, test_ratio: float ,is_pretrain: bool=True):
    """
    generate data for link prediction task (inductive & transductive settings)
    :param dataset_name: str, dataset name
    :param val_ratio: float, validation data ratio
    :param test_ratio: float, test data ratio
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, (Data object)
    """
    # Load data and train val test split
    if dataset_name=='bot22' and not is_pretrain:
        NODE_FEAT_DIM = EDGE_FEAT_DIM = 778
        graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
        edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
        node_raw_features = np.load('./processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name)) 
    
    elif dataset_name=='bot22' and  is_pretrain:
        NODE_FEAT_DIM = 778
        EDGE_FEAT_DIM = 778
        graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
        edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
        node_raw_features = np.load('./processed_data/{}/ml_{}_node_pretrained.npy'.format(dataset_name, dataset_name)) 
    elif dataset_name in ['bot']:
        NODE_FEAT_DIM = 778
        EDGE_FEAT_DIM = 778
        graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
        edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
        node_raw_features = np.load('./processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))     
        print(node_raw_features.shape)
    elif dataset_name=='yelp' :
        NODE_FEAT_DIM = 300
        EDGE_FEAT_DIM = 64
        graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
        edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
        node_raw_features = np.load('./processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))   
    else:
        NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
        graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
        edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
        node_raw_features = np.load('./processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))
    assert NODE_FEAT_DIM >= node_raw_features.shape[1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
    assert EDGE_FEAT_DIM >= edge_raw_features.shape[1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'
    # padding the features of edges and nodes to the same dimension (172 for all the datasets)
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)

    assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[1], 'Unaligned feature dimensions after feature padding!'

    # get the timestamp of validate and test set
    val_time, test_time = list(np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))

    src_node_ids = graph_df.u.values.astype(np.longlong)
    dst_node_ids = graph_df.i.values.astype(np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.idx.values.astype(np.longlong)
    if dataset_name=='bot' or dataset_name== 'bot22' or dataset_name=='dsub' or dataset_name=='dgraph' or dataset_name =='yelp':
        labels = graph_df.label_u.values
    else :
        labels = graph_df.label.values

    labels_time = graph_df.last_u_ts.values
# labels have two lists, for this task we do not use it thus we only take one

    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times, 
                     edge_ids=edge_ids, labels=labels, labels_time = labels_time)

    # the setting of seed follows previous works
    random.seed(2020)

    # union to get node set
    node_set = set(src_node_ids) | set(dst_node_ids)
    num_total_unique_node_ids = len(node_set)

    # compute nodes which appear at test time
    test_node_set = set(src_node_ids[node_interact_times > val_time]).union(set(dst_node_ids[node_interact_times > val_time]))
    # sample nodes which we keep as new nodes (to test inductiveness), so then we have to remove all their edges from training
    new_test_node_set = set(random.sample(list(test_node_set), min(1, len(test_node_set))))  # 作为示例，仅选择最多1个节点
    # new_test_node_set = set(random.sample(list(test_node_set), int(0.1 * num_total_unique_node_ids)))

    # mask for each source and destination to denote whether they are new test nodes
    new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
    new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

    # mask, which is true for edges with both destination and source not being new test nodes (because we want to remove all edges involving any new test node)
    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

    # for train data, we keep edges happening before the validation time which do not involve any new node, used for inductiveness
    train_mask = np.logical_and(node_interact_times <= val_time, observed_edges_mask)

    train_data = Data(src_node_ids=src_node_ids[train_mask], dst_node_ids=dst_node_ids[train_mask],
                      node_interact_times=node_interact_times[train_mask],
                      edge_ids=edge_ids[train_mask], labels=labels[train_mask], labels_time = labels_time[train_mask])

    # define the new nodes sets for testing inductiveness of the model
    train_node_set = set(train_data.src_node_ids).union(train_data.dst_node_ids)
    assert len(train_node_set & new_test_node_set) == 0
    # new nodes that are not in the training set
    new_node_set = node_set - train_node_set

    val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
    test_mask = node_interact_times > test_time

    # new edges with new nodes in the val and test set (for inductive evaluation)
    edge_contains_new_node_mask = np.array([(src_node_id in new_node_set or dst_node_id in new_node_set)
                                            for src_node_id, dst_node_id in zip(src_node_ids, dst_node_ids)])
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

    # validation and test data
    val_data = Data(src_node_ids=src_node_ids[val_mask], dst_node_ids=dst_node_ids[val_mask],
                    node_interact_times=node_interact_times[val_mask], edge_ids=edge_ids[val_mask], 
                    labels=labels[val_mask],labels_time = labels_time[val_mask])

    test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask],
                     node_interact_times=node_interact_times[test_mask], edge_ids=edge_ids[test_mask], 
                     labels=labels[test_mask], labels_time = labels_time[test_mask])

    # validation and test with edges that at least has one new node (not in training set)
    new_node_val_data = Data(src_node_ids=src_node_ids[new_node_val_mask], dst_node_ids=dst_node_ids[new_node_val_mask],
                             node_interact_times=node_interact_times[new_node_val_mask],
                             edge_ids=edge_ids[new_node_val_mask], labels=labels[new_node_val_mask])

    new_node_test_data = Data(src_node_ids=src_node_ids[new_node_test_mask], dst_node_ids=dst_node_ids[new_node_test_mask],
                              node_interact_times=node_interact_times[new_node_test_mask],
                              edge_ids=edge_ids[new_node_test_mask], labels=labels[new_node_test_mask])

    print("The dataset has {} interactions, involving {} different nodes".format(full_data.num_interactions, full_data.num_unique_nodes))
    print("The training dataset has {} interactions, involving {} different nodes".format(
        train_data.num_interactions, train_data.num_unique_nodes))
    print("The validation dataset has {} interactions, involving {} different nodes".format(
        val_data.num_interactions, val_data.num_unique_nodes))
    print("The test dataset has {} interactions, involving {} different nodes".format(
        test_data.num_interactions, test_data.num_unique_nodes))
    print("The new node validation dataset has {} interactions, involving {} different nodes".format(
        new_node_val_data.num_interactions, new_node_val_data.num_unique_nodes))
    print("The new node test dataset has {} interactions, involving {} different nodes".format(
        new_node_test_data.num_interactions, new_node_test_data.num_unique_nodes))
    print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(len(new_test_node_set)))

    all_nodes_set = set(full_data.src_node_ids).union(full_data.dst_node_ids)

    # 联合训练、验证、测试数据集中的节点
    combined_nodes_set = train_node_set.union(set(val_data.src_node_ids)).union(set(val_data.dst_node_ids)).union(set(test_data.src_node_ids)).union(set(test_data.dst_node_ids))

    # 检查是否有节点被遗漏
    assert all_nodes_set == combined_nodes_set, "Some nodes are missing in the datasets."
    
    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data


def get_node_classification_data(dataset_name: str, val_ratio: float, test_ratio: float,is_pretrained:bool=False, new_spilt: bool=True):
    """
    generate data for node classification task
    :param dataset_name: str, dataset name
    :param val_ratio: float, validation data ratio
    :param test_ratio: float, test data ratio
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, (Data object)
    """
    # Load data and train val test split
    if dataset_name=='bot22' and not is_pretrained:
        NODE_FEAT_DIM = EDGE_FEAT_DIM = 778
        graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
        edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
        node_raw_features = np.load('./processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name)) 
    elif dataset_name=='bot22' and  is_pretrained:
        NODE_FEAT_DIM = 778
        EDGE_FEAT_DIM = 778
        graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
        edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
        node_raw_features = np.load('./processed_data/{}/ml_{}_node_pretrained.npy'.format(dataset_name, dataset_name)) 
    elif dataset_name =='bot':
        NODE_FEAT_DIM = EDGE_FEAT_DIM = 778
        graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
        edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
        node_raw_features = np.load('./processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))
    elif dataset_name=='yelp' :
        NODE_FEAT_DIM = 300
        EDGE_FEAT_DIM = 64
        graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
        edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
        node_raw_features = np.load('./processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))   
    else:
        NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
        graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
        edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
        node_raw_features = np.load('./processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))
    assert NODE_FEAT_DIM >= node_raw_features.shape[1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
    assert EDGE_FEAT_DIM >= edge_raw_features.shape[1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'
    # padding the features of edges and nodes to the same dimension (172 for all the datasets)
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)

    assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[1], 'Unaligned feature dimensions after feature padding!'
    double_way_datasets = ['bot','bot22','dgraph','dsub']
    src_node_ids = graph_df.u.values.astype(np.longlong)
    dst_node_ids = graph_df.i.values.astype(np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.idx.values.astype(np.longlong)
    if dataset_name in double_way_datasets:
        label1 = graph_df.label_u.values
        label2 = graph_df.label_i.values
        labels_time1 = graph_df.last_u_ts.values
        labels_time2 = graph_df.last_i_ts.values
        labels=[label1,label2]
        labels_time = [labels_time1,labels_time2]
    else:
        labels=graph_df.label.values
        labels_time = graph_df.last_timestamp.values
    # The setting of seed follows previous works
    random.seed(2020)
    if isinstance(labels, list):
        all_labels = np.concatenate(labels)
    else:
        all_labels = labels

    num_classes = len(np.unique(all_labels))
    if new_spilt:
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
            train_mask = node_interact_times <= val_time
            val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
            test_mask = node_interact_times > test_time
            train_nodes_mask = merged_node_interact_times <= val_time
            train_nodes = merged_ids[train_nodes_mask & mask].astype(int)
 
        else :
            mask = node_interact_times == labels_time
            ground_truth_times = node_interact_times[mask]
            val_time, test_time = list(np.quantile(ground_truth_times, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))
            train_mask = node_interact_times <= val_time
            val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
            test_mask = node_interact_times > test_time
            train_nodes_mask = node_interact_times <= val_time
            train_nodes = merged_ids[train_nodes_mask & mask].astype(int)        
        
    else:
        # get the timestamp of validate and test set
        val_time, test_time = list(np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))
        train_mask = node_interact_times <= val_time
        val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
        test_mask = node_interact_times > test_time
    
    train_nodes = np.unique(train_nodes)

    if dataset_name in double_way_datasets:
        full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times, edge_ids=edge_ids, 
                        labels=labels, labels_time = labels_time)
        train_data = Data(src_node_ids=src_node_ids[train_mask], dst_node_ids=dst_node_ids[train_mask],
                        node_interact_times=node_interact_times[train_mask],
                        edge_ids=edge_ids[train_mask], labels=[label1[train_mask],label2[train_mask]], labels_time = [labels_time1[train_mask],labels_time2[train_mask]])
        val_data = Data(src_node_ids=src_node_ids[val_mask], dst_node_ids=dst_node_ids[val_mask],
                        node_interact_times=node_interact_times[val_mask], edge_ids=edge_ids[val_mask], labels=[label1[val_mask],label2[val_mask]],
                        labels_time = [labels_time1[val_mask],labels_time2[val_mask]])
        test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask],
                        node_interact_times=node_interact_times[test_mask], edge_ids=edge_ids[test_mask],labels=[label1[test_mask],label2[test_mask]],
                        labels_time = [labels_time1[test_mask],labels_time2[test_mask]])
    else:
        full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times, edge_ids=edge_ids, 
                         labels=labels, labels_time = labels_time)
        train_data = Data(src_node_ids=src_node_ids[train_mask], dst_node_ids=dst_node_ids[train_mask],
                        node_interact_times=node_interact_times[train_mask],
                        edge_ids=edge_ids[train_mask], labels=labels[train_mask], labels_time = labels_time[train_mask])
        val_data = Data(src_node_ids=src_node_ids[val_mask], dst_node_ids=dst_node_ids[val_mask],
                        node_interact_times=node_interact_times[val_mask], edge_ids=edge_ids[val_mask], labels=labels[val_mask], 
                        labels_time = labels_time[val_mask])
        test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask],
                        node_interact_times=node_interact_times[test_mask], edge_ids=edge_ids[test_mask],labels=labels[test_mask], 
                        labels_time = labels_time[test_mask])
        
    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, train_nodes, num_classes


def get_NcEM_data(dataset_name: str, val_ratio: float, test_ratio: float ,is_pretrained: bool=True, new_spilt: bool=False):
    """
    generate data for Node classificatin task with EM algorithm
    :param dataset_name: str, dataset name
    :param val_ratio: float, validation data ratio
    :param test_ratio: float, test data ratio
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, num_interactions, num_node_features (Data object)
    """
    # Load data and train val test split
    if dataset_name == 'bot22' and not is_pretrained:
        NODE_FEAT_DIM = EDGE_FEAT_DIM = 778
        graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
        edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
        node_raw_features = np.load('./processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name)) 
    elif dataset_name == 'bot22' and  is_pretrained:
        NODE_FEAT_DIM = 778
        EDGE_FEAT_DIM = 778
        graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
        edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
        node_raw_features = np.load('./processed_data/{}/ml_{}_node_pretrained.npy'.format(dataset_name, dataset_name)) 
    elif dataset_name == 'bot' :
        NODE_FEAT_DIM = 778
        EDGE_FEAT_DIM = 778
        graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
        edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
        node_raw_features = np.load('./processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))     
    elif dataset_name == 'yelp' :
        NODE_FEAT_DIM = 300
        EDGE_FEAT_DIM = 64
        graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
        edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
        node_raw_features = np.load('./processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))   
    else:
        NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
        graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
        edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
        node_raw_features = np.load('./processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))
    assert NODE_FEAT_DIM >= node_raw_features.shape[1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
    assert EDGE_FEAT_DIM >= edge_raw_features.shape[1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'
    # padding the features of edges and nodes to the same dimension (172 for all the datasets)
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)

    assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[1], 'Unaligned feature dimensions after feature padding!'

    src_node_ids = graph_df.u.values.astype(np.longlong)
    dst_node_ids = graph_df.i.values.astype(np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.idx.values.astype(np.longlong)

    double_way_datasets = ['bot','bot22','dgraph','dsub','yelp']

    if dataset_name in double_way_datasets :
        label1 = graph_df.label_u.values
        label2 = graph_df.label_i.values
        labels_time1 = graph_df.last_u_ts.values
        labels_time2 = graph_df.last_i_ts.values
        labels=[label1,label2]
        labels_time = [labels_time1,labels_time2]
    else:
        labels=graph_df.label.values
        labels_time = graph_df.last_timestamp.values
    # The setting of seed follows previous works
    random.seed(2020)
    if isinstance(labels, list):
        all_labels = np.concatenate(labels)
    else:
        all_labels = labels

    num_classes = len(np.unique(all_labels))

    if new_spilt:
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
            train_mask = node_interact_times <= val_time
            val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
            test_mask = node_interact_times > test_time
            train_nodes_mask = merged_node_interact_times <= val_time
            train_nodes = merged_ids[train_nodes_mask & mask].astype(int)
            
        else :
            mask = node_interact_times == labels_time
            ground_truth_times = node_interact_times[mask]
            val_time, test_time = list(np.quantile(ground_truth_times, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))
            train_mask = node_interact_times <= val_time
            val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
            test_mask = node_interact_times > test_time
            train_nodes_mask = node_interact_times <= val_time
            train_nodes = merged_ids[train_nodes_mask & mask].astype(int)    
 
    else:
        # get the timestamp of validate and test set
        val_time, test_time = list(np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))
        train_mask = node_interact_times <= val_time
        val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
        test_mask = node_interact_times > test_time

    val_offest = sum(train_mask)
    test_offest = val_offest+sum(val_mask)
    train_nodes = np.unique(train_nodes)

    if dataset_name in double_way_datasets:
        full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times, edge_ids=edge_ids, 
                        labels=labels, labels_time = labels_time)
        train_data = Data(src_node_ids=src_node_ids[train_mask], dst_node_ids=dst_node_ids[train_mask],
                        node_interact_times=node_interact_times[train_mask],
                        edge_ids=edge_ids[train_mask], labels=[label1[train_mask],label2[train_mask]], labels_time = [labels_time1[train_mask],labels_time2[train_mask]])
        val_data = Data(src_node_ids=src_node_ids[val_mask], dst_node_ids=dst_node_ids[val_mask],
                        node_interact_times=node_interact_times[val_mask], edge_ids=edge_ids[val_mask], labels=[label1[val_mask],label2[val_mask]],
                        labels_time = [labels_time1[val_mask],labels_time2[val_mask]])
        test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask],
                        node_interact_times=node_interact_times[test_mask], edge_ids=edge_ids[test_mask],labels=[label1[test_mask],label2[test_mask]],
                        labels_time = [labels_time1[test_mask],labels_time2[test_mask]])
    else:
        full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times, edge_ids=edge_ids, 
                         labels=labels, labels_time = labels_time)
        train_data = Data(src_node_ids=src_node_ids[train_mask], dst_node_ids=dst_node_ids[train_mask],
                        node_interact_times=node_interact_times[train_mask],
                        edge_ids=edge_ids[train_mask], labels=labels[train_mask], labels_time = labels_time[train_mask])
        val_data = Data(src_node_ids=src_node_ids[val_mask], dst_node_ids=dst_node_ids[val_mask],
                        node_interact_times=node_interact_times[val_mask], edge_ids=edge_ids[val_mask], labels=labels[val_mask], 
                        labels_time = labels_time[val_mask])
        test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask],
                        node_interact_times=node_interact_times[test_mask], edge_ids=edge_ids[test_mask],labels=labels[test_mask], 
                        labels_time = labels_time[test_mask])
        
    # print("The dataset has {} interactions, involving {} different nodes".format(full_data.num_interactions, full_data.num_unique_nodes))
    # print("The training dataset has {} interactions, involving {} different nodes".format(
    #     train_data.num_interactions, train_data.num_unique_nodes))
    # print("The validation dataset has {} interactions, involving {} different nodes".format(
    #     val_data.num_interactions, val_data.num_unique_nodes))
    # print("The test dataset has {} interactions, involving {} different nodes".format(
    #     test_data.num_interactions, test_data.num_unique_nodes)) 
    
    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, full_data.num_interactions, NODE_FEAT_DIM, val_offest, test_offest, train_nodes, num_classes
