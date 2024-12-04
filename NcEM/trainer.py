import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import Optimizer
import shutil
import os

from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from utils.EarlyStopping import EarlyStopping


def get_optimizer(name, parameters, lr, weight_decay=0):
    if name == 'Adam':
        optimizer = torch.optim.Adam(
            params=parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'SGD':
        optimizer = torch.optim.SGD(
            params=parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'RMSprop':
        optimizer = torch.optim.RMSprop(
            params=parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Wrong value for optimizer {name}!")

    return optimizer


def change_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


class Trainer(object):
    def __init__(self, args, model, model_name, logger):
        self.args = args
        self.model = model
        self.model_name = model_name
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.parameters = [p for p in self.model.parameters()]
        logger.info(f'model -> {model}')
        logger.info(f'model name: {model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                    f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')
        self.criterion.to(args.device)
        self.optimizer = get_optimizer(
            self.args.optimizer, self.parameters, self.args.learning_rate, self.args.weight_decay)

    def reset(self):
        self.model.reset()
        self.optimizer = get_optimizer(
            self.args.optimizer, self.parameters, self.args.learning_rate, self.args.weight_decay)

    def save(self, filename):
        params = {
            'model': self.model.state_dict(),
            'optim': self.optimizer.state_dict()
        }
        try:
            torch.save(params, filename)
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optim'])
