import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms import ToPILImage
from datasets.common import *

class OfficeHome(Dataset):
    domains = ['Art', 'Clipart', 'Product', 'Real World']
    num_classes = 65
    
    def __init__(self, root, domains_to_use=['Art', 'Clipart', 'Product', 'Real World'], data_transform=None, target_transform=None):
        super().__init__()
        assert all(domain in OfficeHome.domains for domain in domains_to_use)
        
        dataset_path = os.path.join(root, self.__class__.__name__, 'OfficeHomeDataset_10072016')
        assert os.path.exists(dataset_path)
        self.environments = [
            DomainDataset(domain, dataset_path, data_transform=data_transform, target_transform=target_transform)
            if domain in domains_to_use else () for domain in OfficeHome.domains
        ]
        self.num_datapoints = sum(len(d) for d in self.environments)
    
    def __getitem__(self, idx):
        for d_idx, d in enumerate(self.environments):
            if idx < d.__len__():
                x, y = d.__getitem__(idx)
                env_idx = d_idx
                break
            else:
                idx -= d.__len__()
        return x, (y, env_idx)
    
    def __len__(self):
        return self.num_datapoints