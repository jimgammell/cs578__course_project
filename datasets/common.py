# Download script taken from the DomainBed source code:
#   https://github.com/facebookresearch/DomainBed/blob/main/domainbed/scripts/download.py

import gdown
from zipfile import ZipFile
import tarfile
import os
from collections import OrderedDict
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms import ToPILImage
import torch
from torch.utils.data import Dataset

def download_file(url, dest):
    if not os.path.exists(dest):
        gdown.download(url, dest, quiet=False)

def download_and_extract_dataset(url, dest, remove=True):
    download_file(url, dest)
    if dest.endswith('.tar.gz'):
        tar = tarfile.open(dest, 'r:gz')
        tar.extractall(os.path.dirname(dest))
        tar.close()
    elif dest.endswith('.tar'):
        tar = tarfile.open(dest, 'r:')
        tar.extractall(os.path.dirname(dest))
        tar.close()
    elif dest.endswith('.zip'):
        zf = ZipFile(dest, 'r')
        zf.extractall(os.path.dirname(dest))
        zf.close()
    if remove:
        os.remove(dest)

class MultiDomainDataset(Dataset):
    def __init__(self, root, domains_to_use, all_domains, url=None, download_extension=None, download=True, data_transform=None, target_transform=None):
        super().__init__()
        
        self.domains_to_use = domains_to_use
        self.data_transform = data_transform
        self.target_transform = target_transform
        
        assert all(domain in os.listdir(root) for domain in domains_to_use)
        self.environments = [
            DomainDataset(domain, root, data_transform=data_transform, target_transform=target_transform)
            if domain in domains_to_use else () for domain in all_domains#self.domains_to_use
        ]
        self.num_datapoints = sum(len(d) for d in self.environments)
        self.classes = list([env for env in self.environments if isinstance(env, DomainDataset)][0].data_files.keys())
        
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
    
    def get_domain_name(self, idx):
        return self.domains_to_use[idx]
    
    def get_class_name(self, idx):
        return list(self.environments[0].data_files.keys())[idx]
        
class DomainDataset(Dataset):
    def __init__(self, domain, root, data_transform=None, target_transform=None):
        super().__init__()
        
        self.dataset_path = os.path.join(root, domain)
        assert os.path.exists(self.dataset_path)
        self.data_files = OrderedDict()
        for class_name in sorted(os.listdir(self.dataset_path)):
            self.data_files[class_name] = [f for f in os.listdir(os.path.join(self.dataset_path, class_name))]
        self.total_datapoints = sum(len(item) for item in self.data_files.values())
        self.to_pil_image = ToPILImage()
        self.data_transform = data_transform
        self.target_transform = target_transform
        
    def __getitem__(self, idx):
        for y, (class_name, class_files) in enumerate(self.data_files.items()):
            if idx < len(class_files):
                x = read_image(os.path.join(self.dataset_path, class_name, class_files[idx]))
                x = self.to_pil_image(x)
                break
            else:
                idx -= len(class_files)
        if self.data_transform is not None:
            x = self.data_transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y
    
    def __len__(self):
        return self.total_datapoints