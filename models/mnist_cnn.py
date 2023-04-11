# As used in the DomainBed code:
#   https://github.com/facebookresearch/DomainBed/blob/main/domainbed/networks.py

import numpy as np
import torch
from torch import nn

class MnistCnn(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(8, 128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(8, 128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(8, 128),
            nn.GlobalPool2d(pool_fn=torch.mean)
        )
        
    def forward(self, x):
        return self.model(x)