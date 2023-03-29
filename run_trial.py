import numpy as np
import torch
from torch import nn, optim
import resnet

def train_erm_fe(
    dataset=datasets.ColoredMNIST,
    model_constructor=resnet.Classifier,
    model_kwargs={},
    optimizer_constructor=optim.Adam,
    optimizer_kwargs={'lr': 2e-4},
    loss_fn_constructor=nn.CrossEntropy,
    loss_fn_kwargs={},
    device=None,
    

def run_trial(kwargs):
    