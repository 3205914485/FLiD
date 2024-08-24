from torch import nn
from torch.nn import functional as F
import torch
import math
import numpy as np
import copy

class MessageFunction(nn.Module):
  """
  Module which computes the message for a given interaction.
  """

  def compute_message(self, raw_messages):
    return None


class MLPMessageFunction(MessageFunction):
  def __init__(self, raw_message_dimension, message_dimension):
    super(MLPMessageFunction, self).__init__()

    self.mlp = self.layers = nn.Sequential(
      nn.Linear(raw_message_dimension, raw_message_dimension // 2),
      nn.ReLU(),
      nn.Linear(raw_message_dimension // 2, message_dimension),
    )

  def compute_message(self, raw_messages):
    messages = self.mlp(raw_messages)

    return messages


class IdentityMessageFunction(MessageFunction):

  def compute_message(self, raw_messages):
  
    return raw_messages
  def resetparameters(self):
    return None
  
  
class SAFunction(MessageFunction):
  def __init__(self,input_dim,out_dim, num_heads,hidden_dim,drop_out):
    super(SAFunction,self).__init__()
    self.num_att_heads=num_heads
    self.att_head_size=int(hidden_dim/num_heads)
    self.all_hidden_size=hidden_dim
    self.query=nn.Linear(input_dim,self.all_hidden_size)
    self.key=nn.Linear(input_dim,self.all_hidden_size)
    self.value=nn.Linear(input_dim,self.all_hidden_size)
    self.dense=nn.Linear(hidden_dim,out_dim)
    self.redimconv=nn.Linear(input_dim,out_dim)
    self.layernorm=nn.LayerNorm(out_dim,eps=1e-12)
    self.out_dropout=nn.Dropout(drop_out)
    self.resetparameters()
  def resetparameters(self):
    torch.nn.init.kaiming_uniform_(self.query.weight.data)
    torch.nn.init.kaiming_uniform_(self.key.weight.data)
    torch.nn.init.kaiming_uniform_(self.value.weight.data)
    torch.nn.init.kaiming_uniform_(self.dense.weight.data)
    torch.nn.init.kaiming_uniform_(self.redimconv.weight.data)
    
  def transpose_for_scores(self,x):
    new_x_shape=x.size()[:-1]+(self.num_att_heads,self.att_head_size)
    x=x.view(*new_x_shape)
    return x.permute(1,0,2)
    
  def compute_message(self,input):
    mixed_q_l=self.query(input)
    mixed_k_l=self.key(input)
    mixed_v_l=self.value(input)
    ql=self.transpose_for_scores(mixed_q_l)
    kl=self.transpose_for_scores(mixed_k_l)
    vl=self.transpose_for_scores(mixed_v_l)
    att_scores=torch.matmul(ql,kl.transpose(-1,-2))
    att_scores=att_scores/math.sqrt(self.att_head_size)
    att_probs=nn.Softmax(dim=-1)(att_scores)
    context=torch.matmul(att_probs,vl)
    context_layer=context.permute(1,0,2).contiguous()
    new_contest_layer_shape=context_layer.size()[:-2]+(self.all_hidden_size,)
    context_layer=context_layer.view(*new_contest_layer_shape)
    hidden_states=self.out_dropout(context_layer)
    hidden_states=self.dense(hidden_states)
    hidden_states=self.layernorm(hidden_states+self.redimconv(input))
    return hidden_states

  def forward(self,input):
    mixed_q_l=self.query(input)
    mixed_k_l=self.key(input)
    mixed_v_l=self.value(input)
    ql=self.transpose_for_scores(mixed_q_l)
    kl=self.transpose_for_scores(mixed_k_l)
    vl=self.transpose_for_scores(mixed_v_l)
    att_scores=torch.matmul(ql,kl.transpose(-1,-2))
    att_scores=att_scores/math.sqrt(self.att_head_size)
    att_probs=nn.Softmax(dim=-1)(att_scores)
    context=torch.matmul(att_probs,vl)
    context_layer=context.permute(1,0,2).contiguous()
    new_contest_layer_shape=context_layer.size()[:-2]+(self.all_hidden_size,)
    context_layer=context_layer.view(*new_contest_layer_shape)
    hidden_states=self.out_dropout(context_layer)
    hidden_states=self.dense(hidden_states)
    hidden_states=self.layernorm(hidden_states+self.redimconv(input))
    return hidden_states

class SAM(MessageFunction):
  def __init__(self,input_dim,out_dim, num_heads,hidden_dim,drop_out):
    super(SAM,self).__init__()
    self.num_att_heads=num_heads
    self.att_head_size=int(hidden_dim/num_heads)
    self.all_hidden_size=hidden_dim
    self.query=nn.Linear(input_dim,self.all_hidden_size)
    self.key=nn.Linear(input_dim,self.all_hidden_size)
    self.value=nn.Linear(input_dim,self.all_hidden_size)
    self.dense=nn.Linear(hidden_dim,out_dim)
    self.redimconv=nn.Linear(input_dim,out_dim)
    self.layernorm=nn.LayerNorm(out_dim,eps=1e-12)
    self.out_dropout=nn.Dropout(drop_out)
    self.resetparameters()
  def resetparameters(self):
    torch.nn.init.kaiming_uniform_(self.query.weight.data)
    torch.nn.init.kaiming_uniform_(self.key.weight.data)
    torch.nn.init.kaiming_uniform_(self.value.weight.data)
    torch.nn.init.kaiming_uniform_(self.dense.weight.data)
    torch.nn.init.kaiming_uniform_(self.redimconv.weight.data)
    
  def transpose_for_scores(self,x):
    new_x_shape=x.size()[:-1]+(self.num_att_heads,self.att_head_size)
    x=x.view(*new_x_shape)
    return x.permute(0,2,1,3)
    
  def forward(self,input):
    mixed_q_l=self.query(input)
    mixed_k_l=self.key(input)
    mixed_v_l=self.value(input)
    ql=self.transpose_for_scores(mixed_q_l)
    kl=self.transpose_for_scores(mixed_k_l)
    vl=self.transpose_for_scores(mixed_v_l)
    att_scores=torch.matmul(ql,kl.transpose(-1,-2))
    att_scores=att_scores/math.sqrt(self.att_head_size)
    att_probs=nn.Softmax(dim=-1)(att_scores)
    context=torch.matmul(att_probs,vl)
    context_layer=context.permute(0,2,1,3).contiguous()
    new_contest_layer_shape=context_layer.size()[:-2]+(self.all_hidden_size,)
    context_layer=context_layer.view(*new_contest_layer_shape)
    hidden_states=self.out_dropout(context_layer)
    hidden_states=self.dense(hidden_states)
    hidden_states=self.layernorm(hidden_states+self.redimconv(input))
    return hidden_states


class Positional_encoding(nn.Module):
    def __init__(self,d_model,dropout,max_len=10000):
        super(Positional_encoding,self).__init__()
        self.dropout=nn.Dropout(dropout)
        self.d_model=d_model
        positional_encoding=torch.zeros((max_len,d_model))
        position=torch.arange(0,max_len).unsqueeze(1)
        diver_term=torch.exp(torch.arange(0,d_model,2)*-(math.log(10000.0)/d_model))
        positional_encoding[:,0::2]=torch.sin(position*diver_term)
        positional_encoding[:,1::2]=torch.cos(position*diver_term)
        pe=positional_encoding.unsqueeze(0)
        self.register_buffer('pe',pe)
    def forward(self,x):
        x=x+self.pe[:,:x.size(0)]
        return self.dropout(x)
    

class Feed_Forward(nn.Module):
    def __init__(self,input_dim,hidden_dim=2048,dropout=0.1):
        super(Feed_Forward,self).__init__()
        self.L1=nn.Linear(input_dim,hidden_dim)
        self.L2=nn.Linear(hidden_dim,input_dim)
        self.dropout=nn.Dropout(dropout)
    
    def forward(self,x):
        output=F.relu((self.L1(x)))
        output=self.L2(output)
        return output
    

class Encoder(nn.Module):
    def __init__(self,layer,n):
        super(Encoder,self).__init__()
        self.layers=clones(layer,n)
        self.norm=nn.LayerNorm(layer.size)
    def forward(self,x):
        for layer in self.layers:
            x=layer(x)
        return self.norm(x)

class Sublayerconnection(nn.Module):
    def __init__(self,size,dropout):
        super(Sublayerconnection,self).__init__()
        self.norm=nn.LayerNorm(size)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x,sublayer):
        return x+self.dropout(sublayer(self.norm(x)))

class Encoderlayer(nn.Module):
    def __init__(self,size,self_attn,feed_forward,dropout):
        super(Encoderlayer,self).__init__()
        self.self_attn=self_attn
        self.feed_forward=feed_forward
        self.sublayer=clones(Sublayerconnection(size,dropout),2)
        self.size=size
    
    def forward(self,x):
        x=self.sublayer[0](x,lambda x: self.self_attn(x))
        return self.sublayer[1](x,self.feed_forward)

def clones(module,n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class Transformer_Encoder(MessageFunction):
    def __init__(self,n=3,d_model=512,d_ff=1024,h=4,dropout=0.1,message_dim=172):
        super(Transformer_Encoder,self).__init__()
        c=copy.deepcopy
        self.attn=SAM(d_model,d_model,h,d_ff,dropout)
        self.ff=Feed_Forward(input_dim=d_model,hidden_dim=d_ff,dropout=dropout)
        self.position=Positional_encoding(d_model=d_model,dropout=dropout)
        self.encoder=Encoder(Encoderlayer(d_model,c(self.attn),c(self.ff),dropout=dropout),n)
        self.fc=nn.Linear(d_model,message_dim)
    def resetparameters(self):
        for p in self.encoder.parameters():
            if p.dim()>1:
                nn.init.xavier_uniform(p)
        for p in self.position.parameters():
            if p.dim()>1:
                nn.init.xavier_uniform(p)
        nn.init.kaiming_uniform_(self.fc.weight.data)
    def compute_message(self,x):
        x=self.position(x)
        x=self.encoder(x)
        return self.fc(x)


def get_message_function(module_type, raw_message_dimension, message_dimension,num_att_heads,hidden_size,drop_out,encoders,dff):
    if module_type == "mlp":
        return MLPMessageFunction(raw_message_dimension, message_dimension)
    elif module_type == "id":
        return IdentityMessageFunction()
    elif module_type =='sa':
        return SAFunction(raw_message_dimension,message_dimension,num_att_heads,hidden_size,drop_out)
    elif module_type =='te':
        return Transformer_Encoder(d_model=raw_message_dimension,message_dim=message_dimension,n=encoders,d_ff=dff)
