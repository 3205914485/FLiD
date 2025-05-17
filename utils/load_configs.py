import argparse
import sys
import torch

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_link_prediction_args(is_evaluation: bool = False):
    """
    get the args for the link prediction task
    :param is_evaluation: boolean, whether in the evaluation process
    :return:
    """
    # arguments
    parser = argparse.ArgumentParser('Interface for the link prediction task')
    
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--prefix',type=str,help='prefix of work')
    parser.add_argument('--end_runs', type=int, default=1, help='number of runs of training ending')
    parser.add_argument('--start_runs', type=int, default=0, help='number of runs of training starting')

    parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='yelp',
                        choices=['oag', 'wikipedia', 'reddit', 'dsub'])
    parser.add_argument('--batch_size', type=int, default=200, help='batch size')
    parser.add_argument('--model_name', type=str, default='TGAT', help='name of the model, note that EdgeBank is only applicable for evaluation',
                        choices=['TGAT', 'TGN', 'TCL', 'GraphMixer', 'DyGFormer'])
    parser.add_argument('--accelerate',default=False,help='wheather use the acceletate')
    parser.add_argument('--gpu', type=int, default=3, help='number of gpu to use')
    parser.add_argument('--num_neighbors', type=int, default=20, help='number of neighbors to sample for each node')
    parser.add_argument('--sample_neighbor_strategy', type=str, default='recent', choices=['uniform', 'recent', 'time_interval_aware'], help='how to sample historical neighbors')
    parser.add_argument('--time_scaling_factor', default=1e-6, type=float, help='the hyperparameter that controls the sampling preference with time interval, '
                        'a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, '
                        'it works when sample_neighbor_strategy == time_interval_aware')
    parser.add_argument('--num_walk_heads', type=int, default=8, help='number of heads used for the attention in walk encoder')
    parser.add_argument('--num_heads', type=int, default=2, help='number of heads used in attention layer')
    parser.add_argument('--num_layers', type=int, default=1, help='number of model layers')
    parser.add_argument('--walk_length', type=int, default=1, help='length of each random walk')
    parser.add_argument('--time_gap', type=int, default=2000, help='time gap for neighbors to compute node features')
    parser.add_argument('--time_feat_dim', type=int, default=100, help='dimension of the time embedding')
    parser.add_argument('--position_feat_dim', type=int, default=172, help='dimension of the position embedding')
    parser.add_argument('--edge_bank_memory_mode', type=str, default='unlimited_memory', help='how memory of EdgeBank works',
                        choices=['unlimited_memory', 'time_window_memory', 'repeat_threshold_memory'])
    parser.add_argument('--time_window_mode', type=str, default='fixed_proportion', help='how to select the time window size for time window memory',
                        choices=['fixed_proportion', 'repeat_interval'])
    parser.add_argument('--patch_size', type=int, default=1, help='patch size')
    parser.add_argument('--channel_embedding_dim', type=int, default=50, help='dimension of each channel embedding')
    parser.add_argument('--max_input_sequence_length', type=int, default=32, help='maximal length of the input sequence of each node')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam', 'RMSprop'], help='name of optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--patience', type=int, default=20, help='patience for early stopping')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='ratio of validation set')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='ratio of test set')
    parser.add_argument('--num_runs', type=int, default=5, help='number of runs')
    parser.add_argument('--test_interval_epochs', type=int, default=1, help='how many epochs to perform testing once')
    parser.add_argument('--negative_sample_strategy', type=str, default='random', choices=['random', 'historical', 'inductive'],
                        help='strategy for the negative edge sampling')
    parser.add_argument('--load_best_configs', action='store_true', default=False, help='whether to load the best configurations')
    parser.add_argument('--mf', type=str, default='te', choices=['te','sa','id','mlp'],help='message_function to use')
    parser.add_argument('--encoders', type=int, default=1, help='num_encoders')
    parser.add_argument('--dff', type=int, default=172, help='dense_size')    
    parser.add_argument('--sa_att_heads', type=int, default=4, help='num_sam_heads')
    parser.add_argument('--sa_hidden_size', type=int, default=512, help='num_sam_hidden_size') 
    parser.add_argument('--N_GPUS',type= int ,default= 8 ,help='Num_gpus for the distributed training')   
    args = parser.parse_args()
    if not args.accelerate:
        try:
            args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
        
        except:
            parser.print_help()
            sys.exit()
    else:
        args.device=None
        
    if args.model_name == 'EdgeBank':
        assert is_evaluation, 'EdgeBank is only applicable for evaluation!'

    return args

def get_node_classification_em_args():
    """
    get the args for the node classification task
    :return:
    """
    # arguments
    parser = argparse.ArgumentParser('Interface for the node classification task With The EM algorithm')

    # Configuration of the experiment
    parser.add_argument('--method', type=str, default='ptcl', choices=['ptcl', 'sem', 'npl', 'ptcl_2d'],help='Which method to be used to train')
    parser.add_argument('--double_way_datasets', type=list, default = ['dsub','oag'])
    parser.add_argument('--prefix',type=str, default='test', help='prefix of work')
    parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='wikipedia', choices=['oag', 'reddit','dsub', 'wikipedia'])
    parser.add_argument('--batch_size', type=int, default=200, help='batch size')
    parser.add_argument('--mmodel_name', type=str, default='TGAT', help='name of the model of dyg backbone',
                        choices=['TGAT', 'TGN','TCL', 'GraphMixer', 'DyGFormer'])
    parser.add_argument('--emodel_name', type=str, default='mlp', help='name of the model of decoder',
                        choices=['mlp','mlp_bn'])    
    parser.add_argument('--gpu', type=int, default=0, help='number of gpu to use')
    parser.add_argument('--save_pseudo_labels', type=int, default=0, help='Whether save the pseudo labels')   
    parser.add_argument('--mode', type=str, default='ps', choices=['ps','gt'], help='which label to use')   
    
    #training settings:
    
    parser.add_argument('--num_neighbors', type=int, default=20, help='number of neighbors to sample for each node')
    parser.add_argument('--sample_neighbor_strategy', type=str, default='recent', choices=['uniform', 'recent', 'time_interval_aware'], help='how to sample historical neighbors')
    parser.add_argument('--time_scaling_factor', default=1e-6, type=float, help='the hyperparameter that controls the sampling preference with time interval, '
                        'a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, '
                        'it works when sample_neighbor_strategy == time_interval_aware')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--new_spilt', type=bool, default=True, help='new_split for node_classification dataset') 
    parser.add_argument('--num_classes', type=int, default=2, help='classes')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam', 'RMSprop'], help='name of optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='ratio of validation set')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='ratio of test set')
    parser.add_argument('--test_interval_epochs', type=int, default=1, help='how many epochs to perform testing once')
    parser.add_argument('--load_best_configs', action='store_true', default=False, help='whether to load the best configurations')
    parser.add_argument('--end_runs', type=int, default=1, help='number of runs of training ending')
    parser.add_argument('--start_runs', type=int, default=0, help='number of runs of training starting')
    # warmup:
    parser.add_argument('--warmup_e_train', type=int, default=1, help='Whether Train the warmup E model')
    parser.add_argument('--warmup_m_train', type=int, default=1, help='Whether Train the warmup M model')
    parser.add_argument('--num_epochs_e_warmup', type=int, default=1, help='number of epochs of warmup for E step(LinkPrediction)')
    parser.add_argument('--num_epochs_m_warmup', type=int, default=2, help='number of epochs of warmup for M step(NodeClassification)')
    parser.add_argument('--mw_patience', type=int, default=20, help='patience specific for m_warmup')   

    # EM-Iter settings:
    parser.add_argument('--ps_filter', type=str, default='none', help='Whether filter the pseudo labels by entropy or probability')
    parser.add_argument('--filter_threshold', type=float, default=0.9, help='threshold of pseudo labels filter') 
    parser.add_argument('--alpha', type=float, default=1.0, help='alpha of exp decay of ps mask') 
    parser.add_argument('--negative_weight', type=float, default=1.0, help='negative_weight to make the negative samples consider better')    
    parser.add_argument('--use_ps_back', type=int, default=0, help='Whether update the pseudo labels backwards step by step')
    parser.add_argument('--pseudo_entropy_ws', type=int, default=25, help='Pseudo_entropy window size')    
    parser.add_argument('--pseudo_entropy_th', type=float, default=0.8, help='Pseudo_entropy threshold')
    parser.add_argument('--use_unified', type=int, default=0, help='Whether use the unifed EM train')
    parser.add_argument('--use_transductive', type=int, default=0, help='Whether use the transductive training for E Step') 
    parser.add_argument('--use_inductive', type=int, default=0, help='Whether use the inductive training for E Step') 
    parser.add_argument('--decoder', type=int, default=1, help='num_decoders for training')
    parser.add_argument('--gt_weight', type=float, default=0.5, help='gt_weight to make the gt consider better')
    parser.add_argument('--iter_patience', type=int, default=5, help='patience specific for EM iters loop')    
    parser.add_argument('--patience', type=int, default=15, help='patience for early stopping')
    parser.add_argument('--num_em_iters', type=int, default=1, help='number of EM iters')
    parser.add_argument('--num_iters', type=int, default=30, help='number of iters for npl')
    parser.add_argument('--num_epochs_e_step', type=int, default=1, help='number of epochs of E step')
    parser.add_argument('--num_epochs_m_step', type=int, default=1, help='number of epochs of M step')
    parser.add_argument('--num_epochs_npl', type=int, default=1, help='number of epochs of npl train')
    # Model specific settings:

    parser.add_argument('--num_walk_heads', type=int, default=8, help='number of heads used for the attention in walk encoder')
    parser.add_argument('--num_heads', type=int, default=2, help='number of heads used in attention layer')
    parser.add_argument('--num_layers', type=int, default=1, help='number of model layers')
    parser.add_argument('--walk_length', type=int, default=1, help='length of each random walk')
    parser.add_argument('--time_gap', type=int, default=2000, help='time gap for neighbors to compute node features')
    parser.add_argument('--time_feat_dim', type=int, default=100, help='dimension of the time embedding')
    parser.add_argument('--position_feat_dim', type=int, default=172, help='dimension of the position embedding')
    parser.add_argument('--patch_size', type=int, default=1, help='patch size')
    parser.add_argument('--channel_embedding_dim', type=int, default=50, help='dimension of each channel embedding')
    parser.add_argument('--max_input_sequence_length', type=int, default=32, help='maximal length of the input sequence of each node')
    parser.add_argument('--mf', type=str, default='te', choices=['te','sa','id','mlp'],help='message_function to use')
    parser.add_argument('--encoders', type=int, default=1, help='num_encoders')
    parser.add_argument('--dff', type=int, default=172, help='dense_size')    
    parser.add_argument('--sa_att_heads', type=int, default=4, help='num_sam_heads')
    parser.add_argument('--sa_hidden_size', type=int, default=512, help='num_sam_hidden_size')

    try:
        args = parser.parse_args()
        args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    except:
        parser.print_help()
        sys.exit()

    return args
