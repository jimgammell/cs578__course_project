import numpy as np
import torch
from torch import nn, optim

def apply_mixup_to_data(x, y, alpha):
    lbd = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    x = lbd*x + (1-lbd)*x[index, :]
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lbd

def apply_mixup_to_criterion(criterion, logits, y_a, y_b, lbd):
    return lbd*criterion(logits, y_a) + (1-lbd)*criterion(logits, y_b)

def preprocess_image_for_display(image):
    image = 255.0*(0.5*image+0.5)
    image = image.to(torch.uint8)
    image = image.to(torch.float)
    image = image/255.0
    if len(image.shape) == 3:
        image = image.unsqueeze(1)
    elif image.shape[1] == 2:
        image = torch.cat((image, torch.zeros(
            (image.size(0), 1, image.size(2), image.size(3)), device=image.device, dtype=image.dtype)), dim=1)
    image = image.permute(0, 2, 3, 1)
    image = val(image)
    return image

def unpack_batch(batch, device):
    x, (y, _) = batch
    x, y = x.to(device), y.to(device)
    return x, y

def val(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    else:
        return tensor

def acc(logits, labels):
    logits, labels = val(logits), val(labels)
    return np.mean(np.equal(np.argmax(logits, axis=-1), labels))

def hinge_acc(logits, y):
    logits = val(logits)
    return np.mean(np.equal(np.sign(logits), y))

def covariance_penalty(features):
    features_mn = features.mean(dim=0, keepdims=True)
    features_zc = features - features_mn
    covariance = torch.mm(features_zc.permute(1, 0), features_zc)
    penalty = (covariance - torch.diagonal(covariance)).norm(p=2) / (len(features)**2 - len(features))
    return penalty

def run_epoch(dataloader, step_fn, *step_args, compress_fn=np.nanmean, **step_kwargs):
    rv = {}
    for batch in dataloader:
        step_rv = step_fn(batch, *step_args, **step_kwargs)
        for key, item in step_rv.items():
            if not key in rv.keys():
                rv[key] = []
            rv[key].append(item)
    for key, item in rv.items():
        rv[key] = compress_fn(item)
    return rv