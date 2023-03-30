import random
import os
import numpy as np
import torch
from torch import nn, optim
from train.common import *
from train import single_model, gan

def train_feature_extractor(
    fe_type=None, num_epochs=25, constructor_kwargs=None, epoch_kwargs=None, save_dir=None, random_seed=0):
    assert all(arg is not None for arg in (constructor_kwargs, epoch_kwargs))
    random.seed(random_seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    def save_results(train_rv, val_rv, epoch_idx):
        if save_dir is None:
            return
        train_dir = os.path.join(save_dir, 'results', 'train')
        val_dir = os.path.join(save_dir, 'results', 'validation')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        with open(os.path.join(train_dir, 'epoch_{}.pickle'.format(epoch_idx)), 'wb') as F:
            pickle.dump(train_rv, F)
        with open(os.path.join(val_dir, 'epoch_{}.pickle'.format(epoch_idx)), 'wb') as F:
            pickle.dump(val_rv, F)
        if 'generated_images' in val_dir.keys():
            num_plots = len(val_dir['generated_images'])
            cols = int(np.sqrt(num_plots))
            rows = int(np.sqrt(num_plots))+(1 if int(np.sqrt(num_plots))**2 != num_plots else 0)
            if 'reference_images' in val_dir.keys():
                cols *= 2
            (fig, axes) = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
            if 'reference_images' in val_dir.keys():
                reference_axes = axes[:, :axes.shape[1]//2].flatten()
                generated_axes = axes[:, axes.shape[1]//2:].flatten()
            else:
                generated_axes = axes.flatten()
            frames_dir = os.path.join(save_dir, 'eg_frames')
            os.makedirs(frames_dir, exist_ok=True)
            for ax, image in zip(generated_axes, val_dir['generated_images']):
                cmap = 'binary' if image.shape[0] == 1 else 'plasma'
                ax.imshow(image, cmap=cmap)
                ax.set_xticks([])
                ax.set_yticks([])
            fig.suptitle('Epoch {}'.format(epoch_idx))
            fig.savefig(os.path.join(frames_dir, 'frame_{}.jpg'.format(epoch_idx)), dpi=25)
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
    elif fe_type in ('learned_loss_autoencoder__disc_fe', 'learned_loss_autoencoder__gen_fe'):
        trial_objects = construct_adversarial_models(**constructor_kwargs)
        epoch_fn = learned_loss_autoencoder_epoch
    else:
        assert False
    if epoch_fn is not None:
        for epoch_idx in range(1, num_epochs+1):
            train_rv, val_rv = epoch_fn(**trial_objects, **epoch_kwargs)
            save_results(train_rv, val_rv, epoch_idx)
    fe_save_path = os.path.join('.', 'trained_models', fe_type, 'feature_extractor.pth')
    os.makedirs(fe_save_path, exist_ok=True)
    if fe_type in ('random', 'erm', 'fixed_loss_autoencoder'):
        torch.save(trial_objects['model'].state_dict(), fe_save_path)
    elif fe_type in ('gan', 'learned_loss_autoencoder__disc_fe'):
        torch.save(trial_objects['disc'].state_dict(), fe_save_path)
    elif fe_type == 'learned_loss_autoencoder__gen_fe':
        torch.save(trial_objects['gen'].state_dict(), fe_save_path)

def learned_loss_autoencoder_epoch(
    disc=None, gen=None, disc_optimizer=None, gen_optimizer=None, train_dataloader=None, val_dataloader=None, device=None,
    dnae_noise_magnitude=0.0, auxillary_gen_classifier=False, auxillary_disc_classifier=False):
    assert all(arg is not None for arg in locals())
    train_rv = gan.train_epoch(train_dataloader, gen, gen_opt, disc, disc_opt, device,
                               autoencoder_gen=True, dnae_noise_magnitude=dnae_noise_magnitude,
                               auxillary_gen_classifier=auxillary_gen_classifier, auxillary_disc_classifier=auxillary_disc_classifier)
    val_rv = gen.eval_epoch(val_dataloader, gen, disc, device,
                            autoencoder_gen=True, dnae_noise_magnitude=dnae_noise_magnitude,
                            auxillary_gen_classifier=auxillary_gen_classifier, auxillary_disc_classifier=auxillary_disc_classifier)
    return train_rv, val_rv

def gan_epoch(
    disc=None, gen=None, disc_optimizer=None, gen_optimizer=None, train_dataloader=None, val_dataloader=None, device=None,
    auxillary_disc_classifier=False):
    assert all(arg is not None for arg in locals())
    train_rv = gan.train_epoch(train_dataloader, gen, gen_opt, disc, disc_opt, device,
                               auxillary_disc_classifier=auxillary_disc_classifier)
    val_rv = gan.eval_epoch(val_dataloader, gen, disc, device,
                            auxillary_disc_classifier=auxillary_disc_classifier)
    return train_rv, val_rv

def fixed_loss_autoencoder_epoch(
    model=None, optimizer=None, loss_fn=None, train_dataloader=None, val_dataloader=None, device=None,
    dnae_noise_magnitude=0.0, auxillary_classifier=False):
    assert all(arg is not None for arg in locals())
    train_rv = single_model.train_epoch(train_dataloader, model, optimizer, loss_fn, device,
                           autoencoder=True, dnae_noise_magnitude=dnae_noise_magnitude, auxillary_classifier=auxillary_classifier)
    val_rv = single_model.eval_epoch(val_dataloader, model, loss_fn, device,
                        autoencoder=True, dnae_noise_magnitude=dnae_noise_magnitude, auxillary_classifier=auxillary_classifier)
    return train_rv, val_rv

def erm_epoch(
    model=None, optimizer=None, loss_fn=None, train_dataloader=None, val_dataloader=None, device=None):
    assert all(arg is not None for arg in locals())
    train_rv = single_model.train_epoch(train_dataloader, model, optimizer, loss_fn, device)
    val_rv = single_model.eval_epoch(val_dataloader, model, loss_fn, device)
    return train_rv, val_rv

def construct_random_feature_extractor(
    model_constructor,
    model_kwargs,
    device):
    model = model_constructor(**model_kwargs).to(device)
    return model

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
    device=None):
    assert all(arg is not None for arg in locals())
    model = model_constructor(**model_kwargs).to(device)
    optimizer = optimizer_constructor(model.parameters(), **optimizer_kwargs)
    loss_fn = loss_fn_constructor(**loss_fn_kwargs).to(device)
    dataset = dataset_constructor(train=True, **dataset_kwargs)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [4*len(dataset)//5, len(dataset)//5])
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
    device=None):
    assert all(arg is not None for arg in locals())
    disc = disc_constructor(**disc_kwargs).to(device)
    gen = gen_constructor(**gen_kwargs).to(device)
    disc_optimizer = disc_optimizer_constructor(disc.parameters(), **disc_optimizer_kwargs)
    gen_optimizer = gen_optimizer_constructor(gen.parameters(), **gen_optimizer_kwargs)
    dataset = dataset_constructor(train=True, **dataset_kwargs)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [4*len(dataset)//5, len(dataset)//5])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)
    return {'disc': disc, 'gen': gen,
            'disc_optimizer': disc_optimizer, 'gen_optimizer': gen_optimizer,
            'train_dataloader': train_dataloader, 'val_dataloader': val_dataloader,
            'device': device}