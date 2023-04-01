import os
import numpy as np
from datasets import mnist_m, synthetic_digits
import torch
from torchvision import datasets as tv_datasets
from torchvision import transforms
from torch.utils.data import Dataset

class MultiDomainDigits(Dataset):
    domains = ['MNIST', 'SVHN', 'USPS', 'MNIST-M', 'SynDigits']
    num_classes = 10
    
    def __init__(self, domains_to_use=['MNIST', 'SVHN', 'USPS', 'MNIST-M', 'SynDigits'],
                 data_transform=None, target_transform=None, train=True, download=True):
        super().__init__()
        
        assert all(domain in MultiDomainDigits.domains for domain in domains_to_use)
        self.environments = []
        dataset_kwargs = {'target_transform': target_transform, 'download': download}
        if 'MNIST' in domains_to_use:
            mnist_transforms = [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.stack((x, x, x)).view(3, 28, 28)),
                transforms.ToPILImage()
            ]
            if data_transform is not None:
                mnist_transforms.append(data_transform)
            self.environments.append(
                tv_datasets.MNIST(
                    root=os.path.join('.', 'downloads', 'MNIST'),
                    train=train,
                    transform = transforms.Compose(mnist_transforms),
                    **dataset_kwargs
                )
            )
        else:
            self.environments.append(())
        if 'SVHN' in domains_to_use:
            self.environments.append(
                tv_datasets.SVHN(
                    root=os.path.join('.', 'downloads', 'SVHN'),
                    split='train' if train else 'test',
                    transform=data_transform,
                    **dataset_kwargs
                )
            )
        else:
            self.environments.append(())
        if 'USPS' in domains_to_use:
            usps_transforms = [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.stack((x, x, x)).view(3, 16, 16)),
                transforms.ToPILImage()
            ]
            if data_transform is not None:
                usps_transforms.append(data_transform)
            self.environments.append(
                tv_datasets.USPS(
                    root=os.path.join('.', 'downloads', 'USPS'),
                    train=train,
                    transform=transforms.Compose(usps_transforms),
                    **dataset_kwargs
                )
            )
        else:
            self.environments.append(())
        if 'MNIST-M' in domains_to_use:
            self.environments.append(
                mnist_m.MNISTM(
                    root=os.path.join('.', 'downloads', 'MNIST-M'),
                    train=train,
                    transform=data_transform,
                    **dataset_kwargs
                )
            )
        else:
            self.environments.append(())
        if 'SynDigits' in domains_to_use:
            self.environments.append(
                synthetic_digits.SyntheticDigits(
                    root=os.path.join('.', 'downloads', 'SyntheticDigits'),
                    train=train,
                    transform=data_transform,
                    **dataset_kwargs
                )
            )
        else:
            self.environments.append(())
        for idx, dataset in enumerate(self.environments):
            if len(dataset) > 10000:
                indices = np.arange(len(dataset))
                np.random.shuffle(indices)
                indices = indices[:10000]
                self.environments[idx] = torch.utils.data.Subset(dataset, indices)
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