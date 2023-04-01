import numpy as np
import torch
from torch.utils.data import Dataset
from datasets.common import *

class PACS(Dataset):
    
    def __init__(self, root, download=True, domains_to_use=[], data_transform=None, target_transform=None):
        super().__init__()
        assert all(domain in PACS.domains for domain in domains_to_use)
        
        dataset_path = os.path.join(root, self.__class__.__name__)
        if not os.path.exists(dataset_path):
            assert download
            os.makedirs(dataset_path, exist_ok=True)
            download_and_extract_dataset(r'https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd',
                                         os.path.join(dataset_path, 'PACS.zip'))
            