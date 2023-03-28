import numpy as np
import torch
from torch import nn, optim

def preprocess_image_for_display(image):
    image = 255.0*(0.5*image+0.5)
    image = image.to(torch.uint8)
    image = image.to(torch.float)
    image = image/255.0
    image = val(image)
    return image

def unpack_batch(batch, device):
    x, y = batch
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

def run_epoch(dataloader, step_fn, *step_args, compress_fn=np.mean, **step_kwargs):
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