import numpy as np
import torch
from torch import nn
from train.common import *

def train_step(batch, model, optimizer, loss_fn, device):
    x, y = unpack_batch(batch, device)
    
    model.train()
    logits = model(x)
    loss = loss_fn(logits, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    rv = {
        'loss': val(loss),
        'acc': acc(logits, y)
    }
    return rv

def eval_step(batch, model, loss_fn, device):
    x, y = unpack_batch(batch, device)
    
    model.eval()
    logits = model(x)
    loss = loss_fn(logits, y)
    
    rv = {
        'loss': val(loss),
        'acc': acc(logits, y)
    }
    return rv

def train_epoch(dataloader, model, optimizer, loss_fn, device):
    return run_epoch(dataloader, train_step, model, optimizer, loss_fn, device)

def eval_epoch(dataloader, model, loss_fn, device):
    return run_epoch(dataloader, eval_step, model, loss_fn, device)