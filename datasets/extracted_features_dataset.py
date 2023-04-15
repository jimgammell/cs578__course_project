import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from train.common import *

class ExtractedFeaturesDataset(Dataset):
    def __init__(self, raw_dataset, feature_extractor, batch_size=32, device=None):
        if device == None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        feature_extractor = feature_extractor.to(device)
        self.x, self.y, self.y_env = [], [], []
        loss, accuracy = [], []
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
            logits = feature_extractor.classify_features(x)
            loss.append(val(nn.functional.cross_entropy(logits, torch.stack(self.y[-batch_size:]))))
            accuracy.append(acc(logits, torch.stack(self.y[-batch_size:])))
            self.x.extend(torch.unbind(feature_batch))
        loss, accuracy = np.mean(loss), np.mean(accuracy)
        print('\tDone. Loss: {}. Accuracy: {}.')
        self.num_features = feature_extractor.num_features
        self.num_classes = raw_dataset.__class__.num_classes
        self.domains = raw_dataset.__class__.domains
    def __getitem__(self, idx):
        return self.x[idx], (self.y[idx], self.y_env[idx])
    def __len__(self):
        return len(self.x)