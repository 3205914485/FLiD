import torch
import numpy as np
import torch.nn as nn
from models.message_function import get_message_function


class Message(nn.Module):
    def __init__(self, num_events: int, device: str, updater: str = 'gru', message_funciton: str = 'te', dim_messages: int = 172, dim_events: int = 172, windows_size: int = 1, encoders: int = 1, dff: int = 512, drop_out: float = 0.1):
        super(Message, self).__init__()
        self.device = device
        self.num_events = num_events
        self.dim_messages = dim_messages
        self.dim_events = dim_events
        self.windows_size = windows_size
        self.ef = []
        self.window = []
        # if updater=='gru':
        self.updater = nn.GRUCell(
            input_size=dim_messages, hidden_size=dim_messages)
        self.message_function = get_message_function(module_type=message_funciton, raw_message_dimension=dim_events,
                                                     message_dimension=dim_messages, num_att_heads=4, hidden_size=256, encoders=encoders, dff=dff, drop_out=drop_out)
        self.net = torch.zeros(
            [self.num_events, self.dim_messages], device=self.device)
    # def update(self,ef,idx):
    #     self.window.append(idx)
    #     self.ef.append(ef)
    #     if len(self.window)>self.windows_size:
    #         self.window=self.window[-self.windows_size:]
    #     if len(self.ef)>self.windows_size:
    #         self.ef=self.ef[-self.windows_size:]
    #     self.ef_t=self.message_function.compute_message(self.ef[0]).view(-1, self.dim_messages)
    #     for i in range(len(self.ef)-1):
    #         self.ef_t=torch.cat([self.ef_t,self.message_function.compute_message(self.ef[i+1]).view(-1, self.dim_messages)],dim=0)
    #     self.window_tensor=torch.tensor(np.concatenate(self.window,axis=0),device=self.device).view(-1)
    #     self.net[self.window_tensor]=self.updater(self.ef_t,self.net[self.window_tensor].clone())

    # def update(self,ef,idx):
    #     self.window.append(idx)
    #     self.ef.append(ef)
    #     if len(self.window)>self.windows_size:
    #         self.window=self.window[-self.windows_size:]
    #     if len(self.ef)>self.windows_size:
    #         self.ef=self.ef[-self.windows_size:]
    #     self.ef_t=self.message_function.compute_message(torch.cat(self.ef,dim=0).view(-1,self.dim_events)).view(-1,self.dim_messages)
    #     self.window_tensor=torch.tensor(np.concatenate(self.window,axis=0),device=self.device).view(-1).to('cpu')
    #     to_upd=self.net[self.window_tensor].to(self.device)
    #     updated=self.updater(self.ef_t,to_upd)
    #     del self.ef_t
    #     del to_upd
    #     self.net[self.window_tensor]=updated.to('cpu')
    #     del updated
    # self.net.to('cpu')
    def update(self, ef, idx):
        self.window.append(idx)
        self.ef.append(ef)
        if len(self.window) > self.windows_size:
            self.window = self.window[-self.windows_size:]
        if len(self.ef) > self.windows_size:
            self.ef = self.ef[-self.windows_size:]
        self.ef_t = self.message_function.compute_message(torch.cat(
            self.ef, dim=0).view(-1, self.dim_events)).view(-1, self.dim_messages)
        self.window_tensor = torch.tensor(np.concatenate(
            self.window, axis=0), device=self.device).view(-1)
        self.net[self.window_tensor] = self.updater(
            self.ef_t, self.net[self.window_tensor].clone())

    def detach(self):
        self.net.detach_()
        for i in range(len(self.ef)):
            self.ef[i].detach_()
