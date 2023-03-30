import numpy as np
import torch
from torch import nn
from train.common import *

def train_step(batch, model, optimizer, loss_fn, device,
               autoencoder=False, dnae_noise_magnitude=1.0, auxillary_classifier=False):
    x, y = unpack_batch(batch, device)
    model.train()
    
    if autoencoder:
        x_noise = dnae_noise_magnitude*torch.randn_like(x)
        model_features = model.get_features(x + x_noise)
        reconstruction = model.reconstruct_features(model_features)
        reconstruction_loss = loss_fn(reconstruction, x)
        if auxillary_classifier:
            label_logits = model.classify_labels(model_features)
            label_loss = nn.functional.multi_margin_loss(label_logits, y)
            loss = 0.5*reconstruction_loss + 0.5*label_loss
        else:
            label_loss = None
            loss = reconstruction_loss
    else:
        label_loss = reconstruction_loss = None
        logits = model(x)
        loss = loss_fn(logits, y)
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
def eval_step(batch, model, loss_fn, device,
              autoencoder=False, dnae_noise_magnitude=1.0, auxillary_classifier=False):
    x, y = unpack_batch(batch, device)
    model.eval()
    
    if autoencoder:
        x_noise = dnae_noise_magnitude*torch.randn_like(x)
        model_features = model.get_features(x + x_noise)
        reconstruction = model.reconstruct_features(model_features)
        reconstruction_loss = loss_fn(reconstruction, x)
        if auxillary_classifier:
            label_logits = model.classify_labels(model_features)
            label_loss = nn.functional.cross_entropy_loss(label_logits, y)
            loss = 0.5*reconstruction_loss + 0.5*label_loss
        else:
            label_loss = None
            loss = reconstruction_loss
    else:
        label_loss = reconstruction_loss = None
        logits = model(x)
        loss = loss_fn(logits, y)
    
    rv = {}
    if autoencoder:
        rv.update({'reconstruction_loss', val(reconstruction_loss)})
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
               autoencoder_gen=False, dnae_noise_magnitude=1.0, get_sample_images=False, **step_kwargs):
    rv = run_epoch(dataloader, eval_step, model, loss_fn, device,
                   autoencoder_gen=autoencoder_gen, dnae_noise_magnitude=dnae_noise_magnitude, **step_kwargs)
    if get_sample_images:
        assert autoencoder_gen
        if not hasattr(dataloader, gen_input):
            dataloader.gen_input = next(iter(dataloader))[0]
        x = dataloader.gen_input
        fake_x = model(x + dnae_noise_magnitude*torch.randn_like(x))
        rv.update({
            'generated_images': preprocess_image_for_display(fake_x),
            'reference_images': preprocess_image_for_display(x)
        })
    return rv