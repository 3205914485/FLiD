from models.TGAT import TGAT
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.TCL import TCL
from models.GraphMixer import GraphMixer
from models.DyGFormer import DyGFormer
from models.modules import MLPClassifier, MLPClassifier_BN
from utils.utils import convert_to_gpu
from torch import nn
from PTCL.trainer import Trainer


def NPL_init(args, node_raw_features, edge_raw_features, train_data, full_neighbor_sampler, logger):
    r"""
        Initialize NPL Models
        Args:
            args: arguments for the model
            node_raw_features: node raw features
            edge_raw_features: edge raw features
            train_data: training data
            full_neighbor_sampler: neighbor sampler
            logger: logger
        Returns:
            Dirtrainer: NPL trainer
    """
    # create Emodel
    if args.emodel_name == 'TGAT':
        dynamic_backbone = TGAT(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
    elif args.emodel_name in ['TGN']:
        # four floats that represent the mean and standard deviation of source and destination node time shifts in the training data, which is used for JODIE
        src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
            compute_src_dst_node_time_shifts(
                train_data.src_node_ids, train_data.dst_node_ids, train_data.node_interact_times)
        dynamic_backbone = MemoryModel(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                       time_feat_dim=args.time_feat_dim, model_name=args.emodel_name, num_layers=args.num_layers, num_heads=args.num_heads,
                                       dropout=args.dropout, src_node_mean_time_shift=src_node_mean_time_shift, src_node_std_time_shift=src_node_std_time_shift,
                                       dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst, dst_node_std_time_shift=dst_node_std_time_shift, device=args.device)
    elif args.emodel_name == 'TCL':
        dynamic_backbone = TCL(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                               time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads,
                               num_depths=args.num_neighbors + 1, dropout=args.dropout, device=args.device)
    elif args.emodel_name == 'GraphMixer':
        dynamic_backbone = GraphMixer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                      time_feat_dim=args.time_feat_dim, num_tokens=args.num_neighbors, num_layers=args.num_layers, dropout=args.dropout, device=args.device)
    elif args.emodel_name == 'DyGFormer':
        dynamic_backbone = DyGFormer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                     time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                     num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                     max_input_sequence_length=args.max_input_sequence_length, device=args.device)
    else:
        raise ValueError(f"Wrong value for emodel_name {args.emodel_name}!")

        # create Mmodel
    if args.mmodel_name == 'mlp':
        node_classifier = MLPClassifier(
            input_dim=node_raw_features.shape[1], dropout=args.dropout, num_classes=args.num_classes)
    elif args.mmodel_name =='mlp_bn':
        node_classifier = MLPClassifier_BN(
            input_dim=node_raw_features.shape[1], dropout=args.dropout, num_classes=args.num_classes)
    else:
        raise ValueError(f"Wrong value for mmodel_name {args.mmdel_name}!")
     
    dynamic_backbone = convert_to_gpu(dynamic_backbone, device=args.device)
    node_classifier = convert_to_gpu(node_classifier, device=args.device)

    Dirtrainer = Trainer(args=args, model=nn.Sequential(dynamic_backbone,node_classifier),
                       model_name=args.emodel_name, logger=logger)

    return Dirtrainer
