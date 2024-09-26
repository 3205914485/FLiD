# "static training on dynamic datasets with selective models" 
import logging  
import time 
import os
import sys
import torch
import argparse  
import numpy as np
from torch import nn 
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from static.dataset import load_dataset
from static.utils import create_optimizer, convert_to_gpu, set_random_seed
from static.models import RGCN, GCN, GAT, MLP

class Trainer:
    def __init__(self, args):
        self.args = args

        self.node_features, self.labels, self.edge_index, self.edge_type, self.train_data, self.val_data, self.test_data \
            = load_dataset(args.dataset_name, args.val_ratio, args.test_ratio, args.device)
        args.num_relations = len(np.unique(self.edge_type.cpu().numpy()))

    def train(self):
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        args = self.args
        os.makedirs(
            f"./logs/{args.dataset_name}/{args.model_name}/{args.prefix}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(
            f"./logs/{args.dataset_name}/{args.model_name}/{args.prefix}/{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)
        logger.info(f'configuration is {args}')
        for run in range(args.start_runs, args.end_runs):
            set_random_seed(seed=run)
            logger.info(f"********** Run {run + 1} starts. **********")
            best_auc,best_f1,best_acc = 0.0,0.0,0.0
            if args.model_name == 'rgcn':
                self.model = RGCN(    
                    node_dimension=self.node_features.shape[1],
                    embedding_dimension=args.hidden_dim, 
                    num_relations=args.num_relations,
                    num_classes=args.num_classes,
                    dropout=args.dropout
                )
            elif args.model_name == 'gcn':
                self.model = GCN(    
                    node_dimension=self.node_features.shape[1],
                    embedding_dimension=args.hidden_dim, 
                    num_relations=args.num_relations,
                    num_classes=args.num_classes,
                    dropout=args.dropout
                )
            elif args.model_name == 'gat':
                self.model = GAT(    
                    node_dimension=self.node_features.shape[1],
                    embedding_dimension=args.hidden_dim, 
                    num_relations=args.num_relations,
                    num_classes=args.num_classes,
                    dropout=args.dropout
            )
            elif args.model_name == 'mlp':
                self.model = MLP(    
                    node_dimension=self.node_features.shape[1],
                    embedding_dimension=args.hidden_dim, 
                    num_classes=args.num_classes,
                    dropout=args.dropout
            )
            else :
                raise ValueError(f'No model name {args.model_name}exits')
            self.model = convert_to_gpu(self.model,device=args.device)

            self.opt = create_optimizer(self.model, args.optimizer, args.learning_rate, args.weight_decay)
            self.loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.0,args.negative_weight])).float().to(args.device))

            for epoch in range(args.epochs):
                self.model.train()
                self.opt.zero_grad()
                y_pred = self.model(self.node_features, self.edge_index, self.edge_type)
                y_pred = y_pred[self.train_data]
                y_true = self.labels[self.train_data]
                loss = self.loss_func(input=y_pred,target=y_true)
                loss.backward()
                self.opt.step()
                y_pred = torch.softmax(y_pred,dim=1).detach().cpu().numpy()
                auc = roc_auc_score(y_score=y_pred[:,1], y_true=y_true.cpu().numpy())
                binary_y = y_pred[:,1]>args.threshold
                print(sum(binary_y))
                acc = accuracy_score(y_pred=binary_y, y_true=y_true.cpu().numpy())
                f1 = f1_score(y_pred=binary_y, y_true=y_true.cpu().numpy())
                logger.info(
                f'Epoch: {epoch + 1}, learning rate: {self.opt.param_groups[0]["lr"]}, train loss: {loss:.4f}, auc: {auc:.4f}, f1:{f1:.4f}, acc: {acc:.4f}')
                test_auc,test_f1,test_acc = self.test(logger)
                if test_auc > best_auc:
                    best_auc = test_auc
                    best_f1 = test_f1
                    best_acc = test_acc
                logger.info(
                f'Best test auc: {best_auc:.4f}, f1:{best_f1:.4f}, acc: {best_acc:.4f}')                
                
            logger.info(f"********** Run {run + 1} ends. **********\n") 
    def test(self,logger):
        self.model.eval()
        y_val_pred = self.model(self.node_features, self.edge_index, self.edge_type)[self.val_data]
        y_val_true = self.labels[self.val_data]
        val_loss = self.loss_func(input=y_val_pred,target=y_val_true)
        y_val_pred = torch.softmax(y_val_pred,dim=1).detach()
        val_auc = roc_auc_score(y_score=y_val_pred[:,1].cpu().numpy(), y_true=y_val_true.cpu().numpy())
        logger.info(
        f'valid loss: {val_loss:.4f}, auc: {val_auc:.4f}') 
        y_test_pred = self.model(self.node_features, self.edge_index, self.edge_type)[self.test_data]
        y_test_true = self.labels[self.test_data]
        test_loss = self.loss_func(input=y_test_pred,target=y_test_true)
        y_test_pred = torch.softmax(y_test_pred,dim=1).cpu().detach().numpy()
        test_auc = roc_auc_score(y_score=y_test_pred[:,1], y_true=y_test_true.cpu().numpy())
        binary_y = y_test_pred[:,1]>self.args.threshold
        print(sum(binary_y))
        test_acc = accuracy_score(y_pred=binary_y, y_true=y_test_true.cpu().numpy())
        test_f1 = f1_score(y_pred=binary_y, y_true=y_test_true.cpu().numpy())
        test_f1 = f1_score(y_pred=np.argmax(y_test_pred,axis=1), y_true=y_test_true.cpu().numpy())
        logger.info(
        f'test loss: {test_loss:.4f}, auc: {test_auc:.4f}, f1: {test_f1:.4f}, acc: {test_acc:.4f}')     
        return test_auc,test_f1,test_acc    
    

def main():
    parser = argparse.ArgumentParser(description='Parser for static training on dynamic datasets')
    # general
    parser.add_argument('--prefix', type=str, default='Test', help='set training log name')
    parser.add_argument('--dataset_name', type=str, default='dsub', help='the dataset to use')
    parser.add_argument('--model_name', type=str, default='mlp', help='which model to use')
    # model
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--batch_size', type=int, default=200, help='batch size')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--num_classes', type=int, default=2, help='classes')
    parser.add_argument('--negative_weight', type=float, default=1.0, help='negative_weight to make the negative samples consider better') 
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold for negative weights')
    # training
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam', 'RMSprop'], help='name of optimizer')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight decay')

    # data
    parser.add_argument('--new_spilt', type=bool, default=True, help='new_split for node_classification dataset') 
    parser.add_argument('--val_ratio', type=float, default=0.15, help='ratio of validation set')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='ratio of test set')
    parser.add_argument('--test_interval_epochs', type=int, default=1, help='how many epochs to perform testing once')
    parser.add_argument('--end_runs', type=int, default=5, help='number of runs of training ending')
    parser.add_argument('--start_runs', type=int, default=0, help='number of runs of training starting')
    try:
        args = parser.parse_args()
        args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    except:
        parser.print_help()
        sys.exit()
    trainer = Trainer(args)
    trainer.train()

if __name__ == '__main__':
    main()