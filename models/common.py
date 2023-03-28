import numpy as np
import torch
from torch import nn

class GlobalPool2d(nn.Module):
    def __init__(self, pool_fn=torch.mean):
        super().__init__()
        
        self.pool_fn = pool_fn
        
    def forward(self, x):
        out = self.pool_fn(x, dim=(2, 3))
        out = out.view(-1, out.size(1))
        return out
    
    def __repr__(self):
        return self.__class__.__name__+'({})'.format(self.pool_fn.__class__.__name__)