import numpy as np
import torch
from torch import nn
from train.common import *

def add_noise(x, noise_magnitude=1.0):
    return nn.functional.hardtanh(x + noise_magnitude*torch.randn_like(x))

def train_step(batch, model, optimizer, loss_fn, device, feature_covariance_decay=0.0,
               autoencoder=False, dnae_noise_magnitude=0.0, auxillary_classifier=False,
               mixup_alpha=0.0, **kwargs):
    x, y = unpack_batch(batch, device)
    if mixup_alpha != 0.0:
        x, y_a, y_b, lbd = apply_mixup_to_data(x, y, mixup_alpha)
        criterion = lambda logits: apply_mixup_to_criterion(loss_fn, logits, y_a, y_b, lbd)
    else:
        criterion = lambda logits: loss_fn(logits, y)
    model.train()
    
    if autoencoder:
        noisy_x = add_noise(x, dnae_noise_magnitude)
        model_features = model.get_features(noisy_x)
        reconstruction = model.reconstruct_features(model_features)
        reconstruction_loss = nn.functional.mse_loss(reconstruction, x)
        if auxillary_classifier:
            label_logits = model.classify_labels(model_features)
            label_loss = criterion(label_logits)
            loss = 0.5*reconstruction_loss + 0.5*label_loss
        else:
            label_loss = None
            loss = reconstruction_loss
    else:
        label_loss = reconstruction_loss = None
        model_features = model.get_features(x)
        logits = model.classify_features(model_features)
        loss = criterion(logits)
    loss = loss + feature_covariance_decay*covariance_penalty(model_features)
    print(loss)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    rv = {}
    if autoencoder:
        rv.update({'reconstruction_loss': val(reconstruction_loss)})
        if label_loss is not None:
            rv.update({'label_loss': val(label_loss)})
            rv.update({'label_acc': acc(label_logits, y)})
    else:
        rv.update({'loss': val(loss)})
        rv.update({'acc': acc(logits, y)})
    return rv

@torch.no_grad()
def eval_step(batch, model, loss_fn, device, feature_covariance_decay=0.0,
              autoencoder=False, dnae_noise_magnitude=0.0, auxillary_classifier=False, **kwargs):
    x, y = unpack_batch(batch, device)
    model.eval()
    
    if autoencoder:
        noisy_x = add_noise(x, dnae_noise_magnitude)
        model_features = model.get_features(noisy_x)
        reconstruction = model.reconstruct_features(model_features)
        reconstruction_loss = nn.functional.mse_loss(reconstruction, x)
        if auxillary_classifier:
            label_logits = model.classify_labels(model_features)
            label_loss = loss_fn(label_logits, y)
            loss = 0.5*reconstruction_loss + 0.5*label_loss
        else:
            label_loss = None
            loss = reconstruction_loss
    else:
        label_loss = reconstruction_loss = None
        model_features = model.get_features(x)
        logits = model.classify_features(model_features)
        loss = loss_fn(logits, y)
    loss = loss + feature_covariance_decay*covariance_penalty(model_features)
    
    rv = {}
    if autoencoder:
        rv.update({'reconstruction_loss': val(reconstruction_loss)})
        if label_loss is not None:
            rv.update({'label_loss': val(label_loss)})
            rv.update({'label_acc': acc(label_logits, y)})
    else:
        rv.update({'loss': val(loss)})
        rv.update({'acc': acc(logits, y)})
    return rv

def train_epoch(dataloader, model, optimizer, loss_fn, device, **step_kwargs):
    return run_epoch(dataloader, train_step, model, optimizer, loss_fn, device, **step_kwargs)

def eval_epoch(dataloader, model, loss_fn, device, 
               autoencoder=False, dnae_noise_magnitude=0.0, get_sample_images=False, **step_kwargs):
    rv = run_epoch(dataloader, eval_step, model, loss_fn, device,
                   autoencoder=autoencoder, dnae_noise_magnitude=dnae_noise_magnitude, **step_kwargs)
    if get_sample_images:
        assert autoencoder
        if not hasattr(dataloader, 'gen_input'):
            dataloader.gen_input = next(iter(dataloader))[0]
        x = dataloader.gen_input.to(device)
        noisy_x = add_noise(x, dnae_noise_magnitude)
        fake_x = model(noisy_x)
        rv.update({
            'generated_images': preprocess_image_for_display(fake_x),
            'reference_images': preprocess_image_for_display(noisy_x)
        })
    return rv