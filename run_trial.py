import argparse
from collections import OrderedDict
import traceback
import numpy as np
import torch
from torch import nn, optim
from train.train_feature_extractor import train_feature_extractor
from train.train_classifier import evaluate_trained_models
from datasets import mnist_variants, domainbed

FEATURE_EXTRACTOR_CHOICES = [
    'random',
    'imagenet_trained',
    'erm',
    'fixed_loss_autoencoder',
    'learned_loss_autoencoder',
    'gan'
]

LINEAR_CLASSIFIER_CHOICES = [
    'LogisticRegression',
    'SVM',
    'VREx',
    'IRM'
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
        '--train-lc', action='store_true', default=False,
        help='Run the available linear classifier algorithms on all pre-existing feature extractors.'
    )
    parser.add_argument(
        '--fe-trainer', choices=FEATURE_EXTRACTOR_CHOICES+['all'], default=[FEATURE_EXTRACTOR_CHOICES[0]], nargs='*',
        help='Specify the training setup for the feature extractor.'
    )
    parser.add_argument(
        '--lc-trainer', choices=LINEAR_CLASSIFIER_CHOICES+['all'], default='all', nargs='*',
        help='Specify the training setup for the linear classifier.'
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
        '--restart', default=False, action='store_true',
        help='Specify whether to restart training if there are already model checkpoints for the specified configuration.'
    )
    parser.add_argument(
        '--mixup', default=False, action='store_true',
        help='Specify whether to use mixup in training.'
    )
    parser.add_argument(
        '--pretrained', default=False, action='store_true',
        help='Specify whether to use a feature extractor pretrained on ImageNet as a starting point.'
    )
    parser.add_argument(
        '--augment-data', default=False, action='store_true',
        help='Specify whether to use the standard DomainBet dataset augmentations.'
    )
    parser.add_argument(
        '--covariance-decay', default=False, action='store_true',
        help='Specify whether to use feature covariance decay.'
    )
    
    args = parser.parse_args()
    
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
        num_epochs = args.num_epochs
        for dataset in datasets:
            for fe_type in fe_types:
                if fe_type in ['random', 'imagenet_trained']:
                    continue
                for random_seed in seeds:
                    for omitted_domain in range(len(dataset.domains)):
                        try:
                            constructor_kwargs = {'dataset_constructor': dataset}
                            if args.device is not None:
                                constructor_kwargs.update({'device': args.device})
                            print('Training a feature extractor using {} on {} with domain {} held out.'.format(fe_type, dataset, dataset.domains[omitted_domain]))
                            train_feature_extractor(
                                fe_type=fe_type, num_epochs=num_epochs, constructor_kwargs=constructor_kwargs,
                                random_seed=random_seed, omitted_domain=omitted_domain, mixup=args.mixup,
                                pretrained=args.pretrained, augment_data=args.augment_data, covariance_decay=args.covariance_decay
                            )
                            print('Done.')
                            print('\n\n\n')
                        except Exception:
                            traceback.print_exc()
    if args.train_lc:
        evaluate_trained_models(
            classifiers=args.lc_trainer, seeds=args.seed,
            datasets=[DATASET_CHOICES[ds_arg].__name__ for ds_arg in args.dataset],
            overwrite=args.restart, batch_size=32, device=args.device, num_epochs=args.num_epochs
        )
    
if __name__ == '__main__':
    main()