import os
import torch
from torch import nn, optim
from torch.autograd import grad
from models import resnet
from train.common import *
from datasets.extracted_features_dataset import ExtractedFeaturesDataset

# Types of linear classifiers to implement:
#   ERM (logistic regression, SVM)
#   Invariant risk minimization
#   Group distributionally-robust optimization
#   Domain adjusted regression

def train_linear_classifier(dataset_constructor, holdout_domain, fe_type, seed, batch_size=None, device=None):
    if batch_size is None:
        batch_size = 32
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    fe_path = os.path.join('.', 'trained_models', dataset, 'omit_'+holdout_domain, fe_type, 'model__%d.pth'%(seed))
    if not 'MNIST' in dataset.__name__:
        if 'erm' in fe_type:
            fe_model = resnet.PretrainedRN50(dataset_constructor.input_shape, dataset_constructor.num_classes)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    fe_model.load_state_dict(torch.load(fe_path))
    fe_model.eval()
    fe_model.requires_grad = False
    
    train_dataset = dataset_constructor(
        domains_to_use=[dom for dom in dataset_constructor.domains if dom != holdout_domain],
        data_transform=get_default_transform()
    )
    test_dataset = dataset_constructor(
        domains_to_use=[holdout_domain],
        data_transform = get_default_transform()
    )
    train_dataset = extracted_features_dataset(train_dataset, fe_model, batch_size=batch_size, device=device)
    test_dataset = extracted_features_dataset(test_dataset, fe_model, batch_size=batch_size, device=device)
    train_dataloader = torch.utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)
    test_dataloader = torch.utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=10)
    
    classifier = nn.Linear(fe_model.num_features, dataset_constructor.num_classes).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=2e-4)
    
    
def logistic_regression_step(batch, model, optimizer, device):
    x, y, _ = batch
    x, y = x.to(device), y.to(device)
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    rv = {
        'loss': val(loss),
        'acc': acc(logits, y)
    }
    return rv

def svm_step(batch, model, optimizer, device, weight_decay=1e-4):
    x, y, _ = batch
    x, y = x.to(device), y.to(device)
    logits = model(x)
    loss = nn.functional.multi_margin_loss(logits, y) + weight_decay*model.weight.norm(p=2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    rv = {
        'loss': val(loss),
        'acc': acc(logits, y)
    }
    return rv

def vrex_step(batch, model, optimizer, device, penalty_weight=1e-3):
    x, y, y_e = batch
    x, y, y_e = x.to(device), y.to(device), y_e.to(device)
    logits = model(x)
    loss = nn.functional.cross_entropy_loss(logits, y, reduce=False)
    per_env_loss = torch.tensor([
        loss[y_e==y_e0].mean() for y_e0 in torch.unique(y_e)
    ], device=loss.device)
    mean_loss = per_env_loss.mean()
    invariance_penalty = ((per_env_loss - mean_loss) ** 2).mean()
    vrex_loss = mean_loss + penalty_weight*invariance_penalty
    optimizer.zero_grad()
    vrex_loss.backward()
    optimizer.step()
    
    rv = {
        'loss': val(vrex_loss),
        'acc': acc(logits, y)
    }
    return rv

def irm_step(batch, model, optimizer, device, penalty_weight=1e-3):
    x, y, y_e = batch
    x, y, y_e = x.to(device), y.to(device), y_e.to(device)
    logits = model(x)
    empirical_risk, invariance_penalty = 0.0, 0.0
    for y_e0 in torch.unique(y_e):
        logits_e0 = logits[y_e==y_e0]
        y_e0 = y[y_e==y_e0]
        empirical_risk += nn.functional.cross_entropy(logits_e0, y_e0)
        logits_1, logits_2 = logits_e0[::2], logits_e0[1::2]
        y_1, y_2 = y_e0[::2], y_e0[1::2]
        n = np.max((len(logits_1), len(logits_2)))
        logits_1, logits_2 = logits_1[:n], logits_2[:n]
        y_1, y_2 = y_1[:n], y_2[:n]
        scale = torch.tensor(1.0, device=logits.device).requires_grad_()
        loss_1 = nn.functional.cross_entropy(logits_1*scale, y_1)
        loss_2 = nn.functional.cross_entropy(logits_2*scale, y_2)
        grad_1 = grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = grad(loss_2, [scale], create_graph=True)[0]
        invariance_penalty += (grad_1 * grad_2).sum()
    empirical_risk /= len(torch.unique(y_e))
    invariance_penalty /= len(torch.unique(y_e))
    irm_loss = empirical_risk + penalty_weight*invariance_penalty
    optimizer.zero_grad()
    irm_loss.backward()
    optimizer.step()
    
    rv = {
        'loss': val(irm_loss),
        'acc': acc(logits, y)
    }
    return rv

def group_dro_step(batch, model, optimizer, device, q, eta=1e-3):
    x, y, y_e = batch
    x, y, y_e = x.to(device), y.to(device), y_e.to(device)
    logits = model(x)
    losses = torch.zeros(len(q), device=device)
    for y_e0 in torch.unique(y_e):
        losses[y_e0] = nn.functional.cross_entropy(logits[y_e==y_e0], y[y_e==y_e0])
        q[y_e0] *= (eta*losses[y_e0].data).exp()
    q /= q.sum()
    loss = torch.dot(losses, q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    rv = {
        'loss': val(loss),
        'acc': acc(logits, y)
    }
    return rv