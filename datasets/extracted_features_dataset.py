import numpy as np
import torch
from torch.utils.data import Dataset

class ExtractedFeaturesDataset(Dataset):
    def __init__(self, raw_dataset, feature_extractor, batch_size=32, device=None):
        if device == None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        feature_extractor = feature_extractor.to(device)
        self.x, self.y, self.y_env = [], [], []
        for b_idx in range(len(raw_dataset)//batch_size):
            xx = []
            for x_idx in range(b_idx*batch_size, (b_idx+1)*batch_size):
                x, (y, y_env) = raw_dataset[x_idx]
                xx.append(x)
                self.y.append(y)
                self.y_env.append(y_env)
            batch = torch.stack(xx).to(device)
            with torch.no_grad():
                feature_batch = feature_extractor.get_features(batch).detach().cpu()
            self.x.extend(torch.unbind(feature_batch))
        self.num_features = feature_extractor.num_features
        self.num_classes = raw_dataset.__class__.num_classes
        self.domains = raw_dataset.__class__.domains
    def __getitem__(self, idx):
        return self.x[idx], (self.y[idx], self.y_env[idx])
    def __len__(self):
        return len(self.x)