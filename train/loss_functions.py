import numpy as np
import torch
from torch import nn

class EnvironmentAgnosticLossFunction(nn.Module):
    def __init__(self, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn
        
    def forward(self, logits, labels):
        targets, environments = labels
        return self.loss_fn(logits, targets)
    
    def __repr__(self):
        return self.__class__.__name__ + '({})'.format(self.loss_fn)

class ReweightedLossFunction(nn.Module):
    def __init__(self, loss_fn, dataset):
        super().__init__()
        self.loss_fn = loss_fn
        self.loss_fn.reduction = 'none'
        mean_length = np.mean(len(d) for d in dataset.environments)
        self.weights = {
            env_idx: len(d)/mean_length for d in enumerate(dataset.environments)
        }
        
    def forward(self, logits, labels):
        targets, environments = labels
        weights = torch.tensor([self.weights[env_idx] for env_idx in environments])
        loss = self.loss_fn(logits, targets)
        loss = (weights*loss).mean()
        return loss
    
    def __repr__(self):
        return self.__class__.__name__ + '({})'.format(self.loss_fn)