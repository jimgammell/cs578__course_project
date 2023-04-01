import numpy as np
import torch
from torch import nn
from train.common import *

def hinge_loss(logits, y):
    return nn.functional.relu(1-y*logits).mean()

def train_step(batch, gen, gen_opt, disc, disc_opt, device,
               autoencoder_gen=False, dnae_noise_magnitude=1.0,
               feature_covariance_decay=0.0, mixup_alpha=0.0,
               auxillary_gen_classifier=False, auxillary_disc_classifier=False):
    x, y = unpack_batch(batch, device)
    if mixup_alpha != 0.0:
        x, y_a, y_b, lbd = apply_mixup_to_data(x, y, mixup_alpha)
        criterion = lambda logits: apply_mixup_to_criterion(nn.functional.multi_margin_loss, logits, y_a, y_b)
    else:
        criterion = lambda logits: nn.functional.multi_margin_loss(logits, y)
    gen.train()
    disc.train()
    
    # Train discriminator
    with torch.no_grad():
        if autoencoder_gen:
            gen_input_noise = dnae_noise_magnitude*torch.randn_like(x)
            fake_x = gen(x + gen_input_noise)
            pos_input = torch.cat((x, x), dim=1)
            neg_input = torch.cat((x, fake_x), dim=1)
        else:
            z = torch.randn(x.size(0), gen.latent_features, dtype=torch.float, device=device)
            fake_x = gen(z)
            pos_input = x
            neg_input = fake_x
    pos_features = disc.get_features(pos_input)
    neg_features = disc.get_features(neg_input)
    pos_realism_logits = disc.classify_realism(pos_features)
    neg_realism_logits = disc.classify_realism(neg_features)
    disc_realism_loss = 0.5*hinge_loss(pos_realism_logits, 1) + 0.5*hinge_loss(neg_realism_logits, -1)
    if auxillary_disc_classifier:
        pos_label_logits = disc.classify_label(pos_features)
        disc_label_loss = criterion(pos_label_logits)
        disc_loss = 0.5*disc_realism_loss + 0.5*disc_label_loss
    else:
        disc_label_loss = 0.0
        disc_loss = disc_realism_loss
    disc_loss = disc_loss + 0.5*feature_covariance_decay*(covariance_penalty(pos_features) + covariance_penalty(neg_features))
    disc_opt.zero_grad(set_to_none=True)
    disc_loss.backward()
    disc_opt.step()
    
    # Train generator
    if autoencoder_gen:
        gen_input_noise = dnae_noise_magnitude*torch.randn_like(x)
        gen_features = gen.get_features(x + gen_input_noise)
        fake_x = gen.reconstruct_features(gen_features)
        neg_input = torch.cat((x, fake_x), dim=1)
    else:
        z = torch.randn(x.size(0), gen.latent_features, dtype=torch.float, device=device)
        fake_x = gen(z)
        neg_input = fake_x
    neg_features = disc.get_features(neg_input)
    neg_realism_logits = disc.classify_realism(neg_features)
    gen_realism_loss = -neg_realism_logits.mean()
    if auxillary_gen_classifier:
        assert autoencoder_gen
        gen_label_logits = gen.classify_labels(gen_features)
        gen_label_loss = criterion(gen_label_logits)
        gen_loss = 0.5*gen_realism_loss + 0.5*gen_label_loss + feature_covariance_decay*covariance_penalty(gen_features)
    else:
        gen_label_loss = 0.0
        gen_loss = gen_realism_loss
    gen_opt.zero_grad(set_to_none=True)
    gen_loss.backward()
    gen_opt.step()
    
    rv = {
        'disc_loss': val(disc_loss),
        'disc_realism_loss': val(disc_realism_loss),
        'disc_label_loss': val(disc_label_loss),
        'gen_loss': val(gen_loss),
        'gen_realism_loss': val(gen_realism_loss),
        'gen_label_loss': val(gen_label_loss),
        'disc_realism_acc': 0.5*hinge_acc(pos_realism_logits, 1)+0.5*hinge_acc(neg_realism_logits, -1),
        'disc_label_acc': acc(pos_label_logits, y),
        'gen_label_acc': acc(gen_label_logits, y)
    }
    return rv
    
@torch.no_grad()
def eval_step(batch, gen, disc, device,
              autoencoder_gen=False, dnae_noise_magnitude=1.0,
              feature_covariance_decay=0.0,
              auxillary_gen_classifier=False, auxillary_disc_classifier=False):
    x, y = unpack_batch(batch, device)
    gen.eval()
    disc.eval()
    
    if autoencoder_gen:
        gen_input_noise = dnae_noise_magnitude*torch.randn_like(x)
        gen_features = gen.get_features(x + gen_input_noise)
        fake_x = gen.reconstruct_features(gen_features)
        pos_input = torch.cat((x, x), dim=1)
        neg_input = torch.cat((x, fake_x), dim=1)
    else:
        z = torch.randn(x.size(0), gen.latent_features, dtype=torch.float, device=device)
        fake_x = gen(z)
        pos_input = x
        neg_input = fake_x
    pos_features = disc.get_features(pos_input)
    neg_features = disc.get_features(neg_input)
    pos_realism_logits = disc.classify_realism(pos_features)
    neg_realism_logits = disc.classify_realism(neg_features)
    disc_realism_loss = 0.5*hinge_loss(pos_realism_logits, 1) + 0.5*hinge_loss(neg_realism_logits, -1)
    if auxillary_disc_classifier:
        pos_label_logits = disc.classify_label(pos_features)
        disc_label_loss = nn.functional.multi_margin_loss(pos_label_logits, y)
        disc_loss = 0.5*disc_realism_loss + 0.5*disc_label_loss
    else:
        disc_label_loss = 0.0
        disc_loss = disc_realism_loss
    disc_loss = disc_loss + 0.5*feature_covariance_decay(covariance_penalty(pos_features) + covariance_penalty(neg_features))
    gen_realism_loss = -neg_realism_logits.mean()
    if auxillary_gen_classifier:
        assert autoencoder_gen
        gen_label_logits = gen.classify_labels(gen_features)
        gen_label_loss = nn.functional.multi_margin_loss(gen_label_logits, y)
        gen_loss = 0.5*gen_realism_loss + 0.5*gen_label_loss + feature_covariance_decay*covariance_penalty(gen_features)
    else:
        gen_label_loss = 0.0
        gen_loss = gen_realism_loss
    
    rv = {
        'disc_loss': val(disc_loss),
        'disc_realism_loss': val(disc_realism_loss),
        'disc_label_loss': val(disc_label_loss),
        'gen_loss': val(gen_loss),
        'gen_realism_loss': val(gen_realism_loss),
        'gen_label_loss': val(gen_label_loss),
        'disc_realism_acc': 0.5*hinge_acc(pos_realism_logits, 1)+0.5*hinge_acc(neg_realism_logits, -1),
        'disc_label_acc': acc(pos_label_logits, y),
        'gen_label_acc': acc(gen_label_logits, y)
    }
    return rv

def train_epoch(dataloader, gen, gen_opt, disc, disc_opt, device, **step_kwargs):
    return run_epoch(dataloader, train_step, gen, gen_opt, disc, disc_opt, device, **step_kwargs)

@torch.no_grad()
def eval_epoch(dataloader, gen, disc, device,
               get_sample_images=False, autoencoder_gen=False, dnae_noise_magnitude=1.0, **step_kwargs):
    rv = run_epoch(dataloader, eval_step, gen, disc, device,
                   autoencoder_gen=autoencoder_gen, dnae_noise_magnitude=dnae_noise_magnitude, **step_kwargs)
    if get_sample_images:
        if not hasattr(dataloader, gen_input):
            if autoencoder_gen:
                dataloader.gen_input = next(iter(dataloader))[0][:64]
            else:
                dataloader.gen_input = torch.randn(
                    next(iter(dataloader))[0].size(0), gen.latent_features,
                    dtype=torch.float, device=device
                )
        x = dataloader.gen_input.to(device).clone()
        if autoencoder_gen:
            x += dnae_noise_magnitude*torch.randn_like(x)
        x_fake = gen(x)
        rv.update({'generated_images': preprocess_image_for_display(x_fake)})
        if autoencoder_gen:
            rv.update({'reference images': preprocess_image_for_display(x)})
    return rv