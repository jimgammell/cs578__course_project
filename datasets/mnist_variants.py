 # Based on DomainBed implementations here:
 #   https://github.com/facebookresearch/DomainBed/blob/main/domainbed/datasets.py

import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset

def rotate_dataset(data, targets, angle):
    rotation_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Lambda(lambda x: transforms.functional.rotate(
            img=x, angle=angle, fill=(0,), interpolation=transforms.InterpolationMode.BILINEAR)),
        transforms.ToTensor()
    ])
    
    data = data.float()
    data = data.view(-1, 1, 28, 28)
    data = 2*(data/255.0)-1
    for idx, x in enumerate(data):
        data[idx] = rotation_transform(x)
    targets = targets.view(-1).long()
    
    return data, targets
        
def color_dataset(data, targets, spurious_correlation):
    def torch_bernoulli(p, size):
        return (torch.rand(size) < p).float()
    def torch_xor(a, b):
        return (a-b).abs()
    
    targets = (targets < 5).float()
    targets = torch_xor(targets, torch_bernoulli(0.25, len(targets)))
    colors = torch_xor(targets, torch_bernoulli(spurious_correlation, len(targets)))
    data = torch.stack([data, data], dim=1)
    data[torch.tensor(range(len(data))), (1-colors).long(), :, :] *= 0
    data = 2*(data.float()/255.0)-1
    targets = targets.view(-1).long()
    
    return data, targets

def watermark_dataset(data, targets, spurious_correlation):
    def add_square_watermark(image, center, radius):
        for ridx in range(center[0]-radius, center[0]+radius+1):
            for cidx in range(center[1]-radius, center[1]+radius+1):
                if (0 <= ridx < image.shape[1]) and(0 <= cidx < image.shape[2]) and (((center[0]-ridx).abs() == radius) or (np.abs(center[1]-cidx) == radius)):
                    image[:, ridx, cidx] = 1.0
        return image
    def add_plus_watermark(image, center, radius):
        for ridx in range(center[0]-radius, center[0]+radius+1):
            for cidx in range(center[1]-radius, center[1]+radius+1):
                if (0 <= ridx < image.shape[1]) and (0 <= cidx < image.shape[2]) and ((ridx == center[0]) or (cidx == center[1])):
                    image[:, ridx, cidx] = 1.0
        return image
    def torch_bernoulli(p, size):
        return (torch.rand(size) < p).float()
    def torch_xor(a, b):
        return (a-b).abs()
    
    targets = (targets < 5).float()
    targets = torch_xor(targets, torch_bernoulli(0.25, len(targets)))
    watermarks = torch_xor(targets, torch_bernoulli(spurious_correlation, len(targets)))
    data = 2*(data.float()/255.0)-1
    data = data.view(-1, 1, 28, 28)
    for idx, x in enumerate(data):
        watermark_center = torch.randint(2, 26, size=(2,))
        watermark_radius = torch.randint(1, 4, size=(1,))
        if watermarks[idx] == 0:
            data[idx] = add_square_watermark(x, watermark_center, watermark_radius)
        elif watermarks[idx] == 1:
            data[idx] = add_plus_watermark(x, watermark_center, watermark_radius)
    targets = targets.view(-1).long()
    
    return data, targets
        
class MultiEnvironmentDataset(datasets.MNIST):
    def __init__(self, transform_fn, transform_args, *args, data_transform=None, target_transform=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_transform = data_transform
        self.target_transform = target_transform
        
        n_env = len(transform_args)
        len_env = len(self.data) // n_env
        self.env_labels = []
        updated_data, updated_targets = [], []
        for env_idx in range(n_env):
            self.env_labels.extend(len_env*[env_idx])
            data_indices = slice(env_idx*len_env, (env_idx+1)*len_env, 1)
            updated_data_env, updated_targets_env = transform_fn(
                self.data[data_indices], self.targets[data_indices], transform_args[env_idx]
            )
            updated_data.append(updated_data_env)
            updated_targets.append(updated_targets_env)
        self.data = torch.cat(updated_data, dim=0)
        self.targets = torch.cat(updated_targets, dim=0)
            
    def __getitem__(self, idx):
        x, y = self.data[idx], self.targets[idx]
        if self.data_transform is not None:
            x = self.data_transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, (y, self.env_labels[idx])

class ColoredMNIST(MultiEnvironmentDataset):
    input_shape = (2, 28, 28)
    num_classes = 2
    def __init__(self, spurious_correlations, *args, **kwargs):
        super().__init__(*args, color_dataset, spurious_correlations, **kwargs)

class WatermarkedMNIST(MultiEnvironmentDataset):
    input_shape = (1, 28, 28)
    num_classes = 2
    def __init__(self, spurious_correlations, *args, **kwargs):
        super().__init__(*args, watermark_dataset, spurious_correlations, **kwargs)

class RotatedMNIST(MultiEnvironmentDataset):
    input_shape = (1, 28, 28)
    num_classes = 10
    def __init__(self, rotations, *args, **kwargs):
        super().__init__(*args, rotate_dataset, rotations, **kwargs)