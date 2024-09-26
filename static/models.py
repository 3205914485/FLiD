import torch
from torch import nn
from torch_geometric.nn import RGCNConv,FastRGCNConv,GCNConv,GATConv
import torch.nn.functional as F

class RGCN(nn.Module):
    def __init__(self,node_dimension=239,embedding_dimension=128,dropout=0.3,num_relations=4,num_classes=2):
        super(RGCN, self).__init__()
        self.dropout = dropout

        self.linear_relu_input=nn.Sequential(
            nn.Linear(node_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        
        self.rgcn=RGCNConv(embedding_dimension,embedding_dimension,num_relations=num_relations)
        
        self.linear_relu_output1=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2=nn.Linear(embedding_dimension,num_classes)
        
        
        
    def forward(self,x,edge_index,edge_type):
        x=self.linear_relu_input(x)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        # x=self.rgcn(x,edge_index,edge_type)
        x=self.linear_relu_output1(x)
        x=self.linear_output2(x)
            
        return x

class GCN(nn.Module):
    def __init__(self,node_dimension=239,embedding_dimension=128,dropout=0.3,num_relations=4,num_classes=2):
        super(GCN, self).__init__()
        self.dropout = dropout

        self.linear_relu_input=nn.Sequential(
            nn.Linear(node_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        
        self.gcn=GCNConv(embedding_dimension,embedding_dimension)
        
        self.linear_relu_output1=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2=nn.Linear(embedding_dimension,num_classes)
        
        
        
    def forward(self,x,edge_index,edge_type):
        x=self.linear_relu_input(x)
        x=self.gcn(x,edge_index)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.gcn(x,edge_index)
        x=self.linear_relu_output1(x)
        x=self.linear_output2(x)
            
        return x
class GAT(nn.Module):
    def __init__(self,node_dimension=239,embedding_dimension=128,dropout=0.3,num_relations=4,num_classes=2):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.linear_relu_input=nn.Sequential(
            nn.Linear(node_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        
        self.gat=GATConv(embedding_dimension,embedding_dimension)
        
        self.linear_relu_output1=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2=nn.Linear(embedding_dimension,num_classes)
        
        
        
    def forward(self,x,edge_index,edge_type):
        x=self.linear_relu_input(x)
        x=self.gat(x,edge_index)
        x=F.dropout(x,p=self.dropout,training=self.training)
        # x=self.gat(x,edge_index)
        x=self.linear_relu_output1(x)
        x=self.linear_output2(x)
            
        return x            
class MLP(nn.Module):
    def __init__(self, node_dimension=239, embedding_dimension=128, num_classes=2, dropout=0.3):
        super(MLP, self).__init__()
        self.dropout = dropout

        self.linear_relu_input = nn.Sequential(
            nn.Linear(node_dimension, embedding_dimension),
            nn.ReLU()
        )

        self.hidden_layer = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.ReLU()
        )
        
        self.output_layer = nn.Linear(embedding_dimension, num_classes)

    def forward(self, x, edge_index,edge_type):
        x = self.linear_relu_input(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        return x