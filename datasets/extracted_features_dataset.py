import numpy as np
import torch
from torch.utils.data import Dataset

class ExtractedFeaturesDataset(Dataset):
    def __init__(self, raw_dataset, feature_extractor, batch_size=32, device=None):
        if device == None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        feature_extractor = feature_extractor.device()
        self.x, self.y, self.y_env = [], [], []
        for b_idx in range(len(raw_dataset)//batch_size):
            xx = []
            for x_idx in range(b_idx*batch_size, (b_idx+1)*batch_size):
                x, (y, y_env) = raw_dataset[x_idx]
                xx.append(x)
                self.y.append(y)
                self.env.append(y_env)
            batch = torch.stack(xx).device()
            feature_batch = feature_extractor.get_features(batch)
            self.x.extend(torch.unbind(feature_batch))
    def __getitem__(self, idx):
        return self.x[idx], (self.y[idx], self.y_env[idx])
    def __len__(self):
        return len(self.x)