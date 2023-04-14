import random
import os
import time
import pickle
from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt
import torch
import torchvision
from torch import nn, optim
from train.common import *
from train import single_model, gan
from models import resnet
from train.plot_results import *
from datasets.domainbed import get_default_transform, get_augmentation_transform

def train_feature_extractor(
    fe_type=None, num_epochs=25, constructor_kwargs={}, save_dir=None, random_seed=0, omitted_domain=0,
    mixup=False, pretrained=False, augment_data=False, covariance_decay=False
):
    
    def set_default(key, val):
        if not key in constructor_kwargs.keys():
            constructor_kwargs[key] = val
    assert 'dataset_constructor' in constructor_kwargs.keys()
    constructor_kwargs['dataset_kwargs'] = {
        'domains_to_use': [d for idx, d in enumerate(constructor_kwargs['dataset_constructor'].domains) if idx!=omitted_domain]
    }
    if not 'MNIST' in constructor_kwargs['dataset_constructor'].__name__:
        constructor_kwargs['dataset_kwargs'].update({'data_transform': get_augmentation_transform() if augment_data else get_default_transform()})
    set_default('dataloader_kwargs', {'batch_size': 32, 'num_workers': 10})
    set_default('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    if fe_type in ('random', 'erm', 'fixed_loss_autoencoder'):
        if fe_type == 'random':
            set_default('model_constructor', resnet.Classifier)
        elif fe_type == 'fixed_loss_autoencoder':
            set_default('model_constructor', resnet.Autoencoder)
        elif fe_type == 'erm':
            set_default('model_constructor', resnet.PretrainedRN50 if not 'MNIST' in constructor_kwargs['dataset_constructor'].__name__ else resnet.Classifier)
        if 'MNIST' in constructor_kwargs['dataset_constructor'].__name__:
            set_default('model_kwargs', {
                'features': 64,
                'resample_blocks': 2,
                'endomorphic_blocks': 2
            })
        else:
            if fe_type == 'erm' and not 'MNIST' in constructor_kwargs['dataset_constructor'].__name__:
                set_default('model_kwargs', {
                    'pretrained': pretrained
                })
            else:
                set_default('model_kwargs', {
                    'features': 256,
                    'resample_blocks': 3,
                    'endomorphic_blocks': 3
                })
        set_default('optimizer_constructor', optim.Adam)
        set_default('optimizer_kwargs', {'lr': 1e-3, 'weight_decay': 1e-4})
        set_default('loss_fn_constructor', nn.CrossEntropyLoss)
        set_default('loss_fn_kwargs', {})
    elif fe_type in ('learned_loss_autoencoder', 'gan', 'cyclegan'):
        if fe_type in ('learned_loss_autoencoder', 'cyclegan'):
            set_default('gen_constructor', resnet.Autoencoder)
        else:
            set_default('gen_constructor', resnet.Generator)
        set_default('disc_constructor', resnet.Discriminator)
        if 'MNIST' in constructor_kwargs['dataset_constructor'].__name__:
            set_default('disc_kwargs', {
                'features': 64,
                'resample_blocks': 2,
                'endomorphic_blocks': 2
            })
            set_default('gen_kwargs', {
                'features': 64,
                'resample_blocks': 2,
                'endomorphic_blocks': 2
            })
        else:
            set_default('disc_kwargs', {
                'features': 256,
                'resample_blocks': 3,
                'endomorphic_blocks': 3
            })
            set_default('gen_kwargs', {
                'features': 256,
                'resample_blocks': 3,
                'endomorphic_blocks': 3
            })
        set_default('disc_optimizer_constructor', optim.Adam)
        set_default('disc_optimizer_kwargs', {'lr': 5e-5, 'betas': (0.0, 0.999)})
        set_default('gen_optimizer_constructor', optim.Adam)
        set_default('gen_optimizer_kwargs', {'lr': 1e-5, 'betas': (0.0, 0.999)})
    else:
        assert False
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)
    
    fe_type_opt = fe_type
    if mixup:
        fe_type_opt += '_mixup'
    if pretrained:
        fe_type_opt += '_pretrained'
    if augment_data:
        fe_type_opt += '_augmented'
    if covariance_decay:
        fe_type_opt += '_covariance_decay'
    
    if save_dir is None:
        save_dir = os.path.join(
            '.', 'results', constructor_kwargs['dataset_constructor'].__name__,
            'omit_{}'.format(constructor_kwargs['dataset_constructor'].domains[omitted_domain]),
            fe_type_opt, 'trial_{}'.format(random_seed)
        )
    os.makedirs(save_dir, exist_ok=True)
    
    print('Training a feature extractor.')
    print('Save dir: {}'.format(save_dir))
    print('Constructor kwargs: {}'.format(constructor_kwargs))
    
    def save_results(train_rv, val_rv, epoch_idx):
        train_dir = os.path.join(save_dir, 'results', 'train')
        val_dir = os.path.join(save_dir, 'results', 'validation')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        with open(os.path.join(train_dir, 'epoch_{}.pickle'.format(epoch_idx)), 'wb') as F:
            pickle.dump(train_rv, F)
        with open(os.path.join(val_dir, 'epoch_{}.pickle'.format(epoch_idx)), 'wb') as F:
            pickle.dump(val_rv, F)
        if 'generated_images' in val_rv.keys():
            num_plots = len(val_rv['generated_images'])
            cols = int(np.sqrt(num_plots))
            rows = int(np.sqrt(num_plots))+(1 if int(np.sqrt(num_plots))**2 != num_plots else 0)
            if 'reference_images' in val_rv.keys():
                cols *= 2
            (fig, axes) = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
            if 'reference_images' in val_rv.keys():
                reference_axes = axes[:, :axes.shape[1]//2].flatten()
                generated_axes = axes[:, axes.shape[1]//2:].flatten()
            else:
                generated_axes = axes.flatten()
            frames_dir = os.path.join(save_dir, 'eg_frames')
            os.makedirs(frames_dir, exist_ok=True)
            for ax, image in zip(generated_axes, val_rv['generated_images']):
                cmap = 'binary' if image.shape[-1] == 1 else 'plasma'
                ax.imshow(image, cmap=cmap)
                ax.set_xticks([])
                ax.set_yticks([])
            if 'reference_images' in val_rv.keys():
                for ax, image in zip(reference_axes, val_rv['reference_images']):
                    cmap = 'binary' if image.shape[-1] == 1 else 'plasma'
                    ax.imshow(image, cmap=cmap)
                    ax.set_xticks([])
                    ax.set_yticks([])
            fig.suptitle('Epoch {}'.format(epoch_idx))
            fig.savefig(os.path.join(frames_dir, 'frame_{}.jpg'.format(epoch_idx)), dpi=25)
            plt.close('all')
            
    if fe_type == 'random':
        trial_objects = construct_random_feature_extractor(**constructor_kwargs)
        epoch_fn = None
    elif fe_type == 'erm':
        trial_objects = construct_single_model(**constructor_kwargs)
        epoch_fn = erm_epoch
    elif fe_type == 'fixed_loss_autoencoder':
        trial_objects = construct_single_model(**constructor_kwargs)
        epoch_fn = fixed_loss_autoencoder_epoch
    elif fe_type == 'gan':
        trial_objects = construct_adversarial_models(**constructor_kwargs)
        epoch_fn = gan_epoch
    elif fe_type == 'learned_loss_autoencoder':
        trial_objects = construct_adversarial_models(llae=True, **constructor_kwargs)
        epoch_fn = learned_loss_autoencoder_epoch
    else:
        assert False
        
    def print_dict(d):
        for key, item in d.items():
            if not hasattr(item, '__iter__'):
                print('\t\t{}: {}'.format(key, item))

    def get_state_dict(model):
        return {k: v.cpu() for k, v in deepcopy(model).state_dict().items()}
                
    def train_erm_fe():
        best_model = get_state_dict(trial_objects['model'])
        best_train_loss = np.inf
        epochs_without_improvement = 0
        for epoch_idx in range(1, num_epochs+1):
            t0 = time.time()
            train_rv, val_rv = epoch_fn(
                mixup_alpha=1.0 if mixup else 0.0,
                feature_covariance_decay = 1.0 if covariance_decay else 0.0,
                **trial_objects)
            save_results(train_rv, val_rv, epoch_idx)
            print('Epoch {} complete in {} seconds'.format(epoch_idx, time.time()-t0))
            print('\tTrain rv:')
            print_dict(train_rv)
            print('\tVal rv:')
            print_dict(val_rv)
            train_loss = train_rv['loss']
            if train_loss < best_train_loss:
                print('New best model found.')
                best_train_loss = train_loss
                best_model = get_state_dict(trial_objects['model'])
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                print('Epochs without improvement (at current learning rate): {}'.format(epochs_without_improvement))
            if epochs_without_improvement > 5:
                print('Performance gains have saturated. Dividing learning rate by 10.')
                epochs_without_improvement = 0
                for g in trial_objects['optimizer'].param_groups:
                    g['lr'] /= 2.0
        return best_model
    
    def train_fe(num_epochs, trial_objects, save_int_results=False, mixup_alpha=0.0, feature_covariance_decay=0.0):
        rv = {}
        if epoch_fn is not None:
            for epoch_idx in range(1, num_epochs+1):
                t0 = time.time()
                train_rv, val_rv = epoch_fn(
                    mixup_alpha=mixup_alpha,
                    feature_covariance_decay=feature_covariance_decay,
                    **trial_objects)
                for key, item in train_rv.items():
                    if not 'train_'+key in rv.keys():
                        rv['train_'+key] = []
                    rv['train_'+key].append(item)
                for key, item in val_rv.items():
                    if not 'val_'+key in rv.items():
                        rv['val_'+key] = []
                    rv['val_'+key].append(item)
                if save_int_results:
                    save_results(train_rv, val_rv, epoch_idx)
                print('Epoch {} complete in {} seconds'.format(epoch_idx, time.time()-t0))
                print('\tTrain rv:')
                print_dict(train_rv)
                print('\tVal rv:')
                print_dict(val_rv)
        return trial_objects['model'].cpu().state_dict(), rv
    
    def get_optimal_fe_hparams():
        nonlocal constructor_kwargs
        hparams = {'lr': lambda: 10**np.random.uniform(-5, -3.5),
                   'weight_decay': lambda: 10**np.random.uniform(-6, -2)}
        if mixup:
            hparams['mixup_alpha'] = lambda: 10**np.random.uniform(-1, 1)
        else:
            hparams['mixup_alpha'] = lambda: 0.0
        if covariance_decay:
            hparams['feature_covariance_decay'] = lambda: 10**np.random.uniform(-4, 0)
        else:
            hparams['feature_covariance_decay'] = lambda: 0.0
        
        results, hparams_ = [], []
        print('Sweeping hyperparameters for current feature extractor configuration.')
        for trial_idx in range(20):
            print('Starting trial {}...'.format(trial_idx))
            trial_hparams = {hparam_name: get_hparam_fn() for hparam_name, get_hparam_fn in hparams.items()}
            print('\tHyperparameters: {}'.format(trial_hparams))
            constructor_kwargs['optimizer_kwargs']['lr'] = trial_hparams['lr']
            constructor_kwargs['optimizer_kwargs']['weight_decay'] = trial_hparams['weight_decay']
            trial_objects = construct_single_model(**constructor_kwargs)
            _, trial_results = train_fe(10, trial_objects, mixup_alpha=trial_hparams['mixup_alpha'], feature_covariance_decay=trial_hparams['feature_covariance_decay'])
            results.append(trial_results)
            hparams_.append(trial_hparams)
        best_val_acc, best_hparams = -np.inf, None
        for trial_results, trial_hparams in zip(results, hparams_):
            if np.max(trial_results['val_acc']) > best_val_acc:
                best_val_acc = np.max(trial_results['val_acc'])
                best_hparams = trial_hparams
        print('Best hyperparameters found.')
        print('\tBest validation accuracy: {}'.format(best_val_acc))
        print('\tBest hyperparameters: {}'.format(best_hparams))
        constructor_kwargs['optimizer_kwargs']['lr'] = best_hparams['lr']
        constructor_kwargs['optimizer_kwargs']['weight_decay'] = best_hparams['weight_decay']
        trial_objects = construct_single_model(**constructor_kwargs)
        best_model, best_results = train_erm_fe(100, trial_objects, mixup_alpha=best_hparams['mixup_alpha'], feature_covariance_decay=best_hparams['feature_covariance_decay'])
        print('Done.')
        return best_model, best_results
    
    if fe_type == 'erm':
        best_model, _ = get_optimal_fe_hparams()
    else:
        best_model = train_fe()
    
    fe_save_dir = os.path.join(
        '.', 'trained_models', constructor_kwargs['dataset_constructor'].__name__,
        'omit_{}'.format(constructor_kwargs['dataset_constructor'].domains[omitted_domain]), fe_type_opt
    )
    os.makedirs(fe_save_dir, exist_ok=True)
    if fe_type in ('random', 'erm', 'fixed_loss_autoencoder'):
        torch.save(best_model, os.path.join(fe_save_dir, 'model__{}.pth'.format(random_seed)))
    else:
        torch.save(trial_objects['disc'].state_dict(), os.path.join(fe_save_dir, 'disc__{}.pth'.format(random_seed)))
        torch.save(trial_objects['disc_optimizer'].state_dict(), os.path.join(fe_save_dir, 'disc_opt__{}.pth'.format(random_seed)))
        torch.save(trial_objects['gen'].state_dict(), os.path.join(fe_save_dir, 'gen__{}.pth'.format(random_seed)))
        torch.save(trial_objects['gen_optimizer'].state_dict(), os.path.join(fe_save_dir, 'gen_opt__{}.pth'.format(random_seed)))
    if not fe_type in ('random'):
        plot_traces(save_dir)
    if not fe_type in ('random', 'erm'):
        generate_animation(save_dir)

def learned_loss_autoencoder_epoch(
    disc=None, gen=None, disc_optimizer=None, gen_optimizer=None, train_dataloader=None, val_dataloader=None, device=None):
    assert all(arg is not None for arg in locals())
    train_rv = gan.train_epoch(train_dataloader, gen, gen_optimizer, disc, disc_optimizer, device,
                               autoencoder_gen=True, dnae_noise_magnitude=1.0, auxillary_gen_classifier=True)
    val_rv = gan.eval_epoch(val_dataloader, gen, disc, device,
                            autoencoder_gen=True, dnae_noise_magnitude=1.0,
                            auxillary_gen_classifier=True, get_sample_images=True)
    return train_rv, val_rv

def gan_epoch(
    disc=None, gen=None, disc_optimizer=None, gen_optimizer=None, train_dataloader=None, val_dataloader=None, device=None):
    assert all(arg is not None for arg in locals())
    train_rv = gan.train_epoch(train_dataloader, gen, gen_optimizer, disc, disc_optimizer, device,
                               auxillary_disc_classifier=True, mixup_alpha=0.0)
    val_rv = gan.eval_epoch(val_dataloader, gen, disc, device,
                            auxillary_disc_classifier=True, get_sample_images=True)
    return train_rv, val_rv

def fixed_loss_autoencoder_epoch(
    model=None, optimizer=None, loss_fn=None, train_dataloader=None, val_dataloader=None, device=None):
    assert all(arg is not None for arg in locals())
    train_rv = single_model.train_epoch(
        train_dataloader, model, optimizer, loss_fn, device,
        autoencoder=True, dnae_noise_magnitude=1.0,
        auxillary_classifier=True)
    val_rv = single_model.eval_epoch(
        val_dataloader, model, loss_fn, device,
        autoencoder=True, dnae_noise_magnitude=1.0,
        auxillary_classifier=True, get_sample_images=True)
    return train_rv, val_rv

def erm_epoch(
    model=None, optimizer=None, loss_fn=None, train_dataloader=None, val_dataloader=None, device=None, mixup_alpha=0.0, feature_covariance_decay=0.0):
    assert all(arg is not None for arg in locals())
    train_rv = single_model.train_epoch(train_dataloader, model, optimizer, loss_fn, device, mixup_alpha=mixup_alpha, feature_covariance_decay=feature_covariance_decay)
    val_rv = single_model.eval_epoch(val_dataloader, model, loss_fn, device)
    return train_rv, val_rv

def construct_random_feature_extractor(
    dataset_constructor,
    model_constructor,
    model_kwargs,
    device, **kwargs):
    model = model_constructor(dataset_constructor.input_shape, dataset_constructor.num_classes, **model_kwargs).to(device)
    return{'model': model}

def construct_single_model(
    dataset_constructor=None,
    dataset_kwargs=None,
    dataloader_kwargs=None,
    model_constructor=None,
    model_kwargs=None,
    optimizer_constructor=None,
    optimizer_kwargs=None,
    loss_fn_constructor=None,
    loss_fn_kwargs=None,
    device=None, **kwargs):
    assert all(arg is not None for arg in locals())
    model = model_constructor(dataset_constructor.input_shape, dataset_constructor.num_classes, **model_kwargs).to(device)
    optimizer = optimizer_constructor(model.parameters(), **optimizer_kwargs)
    loss_fn = loss_fn_constructor(**loss_fn_kwargs).to(device)
    dataset = dataset_constructor(**dataset_kwargs)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset)-len(dataset)//5, len(dataset)//5])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)
    return {'model': model, 'optimizer': optimizer,
            'loss_fn': loss_fn, 'train_dataloader': train_dataloader,
            'val_dataloader': val_dataloader, 'device': device}
    
def construct_adversarial_models(
    dataset_constructor=None,
    dataset_kwargs=None,
    dataloader_kwargs=None,
    disc_constructor=None,
    disc_kwargs=None,
    gen_constructor=None,
    gen_kwargs=None,
    disc_optimizer_constructor=None,
    disc_optimizer_kwargs=None,
    gen_optimizer_constructor=None,
    gen_optimizer_kwargs=None,
    device=None, llae=False, **kwargs):
    assert all(arg is not None for arg in locals())
    if llae:
        disc_input_shape = [2*dataset_constructor.input_shape[0], *dataset_constructor.input_shape[1:]]
    else:
        disc_input_shape = dataset_constructor.input_shape
    disc = disc_constructor(disc_input_shape, dataset_constructor.num_classes, **disc_kwargs).to(device)
    gen = gen_constructor(dataset_constructor.input_shape, dataset_constructor.num_classes, **gen_kwargs).to(device)
    disc_optimizer = disc_optimizer_constructor(disc.parameters(), **disc_optimizer_kwargs)
    gen_optimizer = gen_optimizer_constructor(gen.parameters(), **gen_optimizer_kwargs)
    dataset = dataset_constructor(**dataset_kwargs)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset)-len(dataset)//5, len(dataset)//5])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)
    return {'disc': disc, 'gen': gen,
            'disc_optimizer': disc_optimizer, 'gen_optimizer': gen_optimizer,
            'train_dataloader': train_dataloader, 'val_dataloader': val_dataloader,
            'device': device}