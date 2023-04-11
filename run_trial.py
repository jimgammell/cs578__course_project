import argparse
from collections import OrderedDict
import traceback
import numpy as np
import torch
from torch import nn, optim
from train.train_feature_extractor import train_feature_extractor
from datasets import mnist_variants, domainbed

FEATURE_EXTRACTOR_CHOICES = [
    'random',
    'erm',
    'fixed_loss_autoencoder',
    'learned_loss_autoencoder',
    'gan'
]

DATASET_CHOICES = OrderedDict([
    ('rotated_mnist', mnist_variants.RotatedMNIST),
    ('colored_mnist', mnist_variants.ColoredMNIST),
    ('watermarked_mnist', mnist_variants.WatermarkedMNIST),
    ('office_home', domainbed.OfficeHome),
    ('vlcs', domainbed.VLCS),
    ('pacs', domainbed.PACS),
    ('sviro', domainbed.Sviro),
    ('domain_net', domainbed.DomainNet),
    ('terra_incognita', domainbed.TerraIncognita)
])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', choices=list(DATASET_CHOICES.keys())+['all'], default=[next(iter(DATASET_CHOICES.keys()))], nargs='*',
        help='Specify the dataset that will be used.'
    )
    parser.add_argument(
        '--train-fe', action='store_true', default=False,
        help='Train a feature extractor.'
    )
    parser.add_argument(
        '--fe-trainer', choices=FEATURE_EXTRACTOR_CHOICES+['all'], default=[FEATURE_EXTRACTOR_CHOICES[0]], nargs='*',
        help='Specify the training setup for the feature extractor.'
    )
    parser.add_argument(
        '--device', default=None,
        help='Specify the device to use for training.'
    )
    parser.add_argument(
        '--num_epochs', default=25, type=int,
        help='Specify the number of epochs to use for training.'
    )
    parser.add_argument(
        '--seed', default=[0], type=int, nargs='*',
        help='Specify the random seed to use for training.'
    )
    parser.add_argument(
        '--restart', default=True, type=bool,
        help='Specify whether to restart training if there are already model checkpoints for the specified configuration.'
    )
    args = parser.parse_args()
    assert args.restart == True
    
    if args.train_fe:
        if 'all' in args.dataset:
            datasets = list(DATASET_CHOICES.values())
        else:
            datasets = [DATASET_CHOICES[ds_arg] for ds_arg in args.dataset]
        if 'all' in args.fe_trainer:
            fe_types = FEATURE_EXTRACTOR_CHOICES
        else:
            fe_types = [fe_arg for fe_arg in args.fe_trainer]
        seeds = [rs_arg for rs_arg in args.seed]
        if args.device is not None:
            constructor_kwargs.update({'device': args.device})
        num_epochs = args.num_epochs
        for dataset in datasets:
            for fe_type in fe_types:
                for random_seed in seeds:
                    for omitted_domain in range(len(dataset.domains)):
                        try:
                            constructor_kwargs = {'dataset_constructor': dataset}
                            print('Training a feature extractor using {} on {} with domain {} held out.'.format(fe_type, dataset, dataset.domains[omitted_domain]))
                            train_feature_extractor(
                                fe_type=fe_type, num_epochs=num_epochs, constructor_kwargs=constructor_kwargs,
                                random_seed=random_seed, omitted_domain=omitted_domain
                            )
                            print('Done.')
                            print('\n\n\n')
                        except Exception:
                            traceback.print_exc()
    
if __name__ == '__main__':
    main()