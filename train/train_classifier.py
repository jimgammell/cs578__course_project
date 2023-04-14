import os
from collections import OrderedDict
from tqdm import tqdm
from copy import deepcopy
import pickle
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import grad
from models import resnet
from train.common import *
from datasets import domainbed
from datasets.extracted_features_dataset import ExtractedFeaturesDataset

# Types of linear classifiers to implement:
#   ERM (logistic regression, SVM)
#   Invariant risk minimization
#   Group distributionally-robust optimization
#   Domain adjusted regression

def get_dataloaders(dataset_constructor, holdout_domain, fe_type, seed, batch_size=None, device=None):
    if batch_size is None:
        batch_size = 32
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fe_path = os.path.join('.', 'trained_models', dataset_constructor.__name__, 'omit_'+holdout_domain, fe_type, 'model__%d.pth'%(seed))
    if not 'MNIST' in dataset_constructor.__name__:
        fe_model = resnet.PretrainedRN50(dataset_constructor.input_shape, dataset_constructor.num_classes,
                                         pretrained=True if fe_type=='imagenet_trained' else False)
    else:
        raise NotImplementedError
    if not fe_type in ['random', 'imagenet_pretrained']:
        fe_model.load_state_dict(torch.load(fe_path, map_location=device))
    fe_model.eval()
    fe_model.requires_grad = False
    
    train_dataset = dataset_constructor(
        domains_to_use=[dom for dom in dataset_constructor.domains if dom != holdout_domain],
        data_transform=domainbed.get_default_transform()
    )
    test_dataset = dataset_constructor(
        domains_to_use=[holdout_domain],
        data_transform=domainbed.get_default_transform()
    )
    train_dataset = ExtractedFeaturesDataset(train_dataset, fe_model, batch_size=32, device=device)
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [len(train_dataset)-len(train_dataset)//5, len(train_dataset)//5]
    )
    test_dataset = ExtractedFeaturesDataset(test_dataset, fe_model, batch_size=32, device=device)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    return train_dataloader, val_dataloader, test_dataloader

class Trainer:
    def __init__(self, num_features, num_classes, device, hparams):
        self.classifier = nn.Linear(num_features, num_classes).to(device)
        self.optimizer = optim.Adam(self.classifier.parameters(), lr=hparams['learning_rate'])
        self.hparams = hparams
        self.device = device
        
    def train_step(self, batch):
        raise NotImplementedError
        
    def eval_step(self, batch):
        x, (y, _) = batch
        x, y = x.to(self.device), y.to(self.device)
        logits = self.classifier(x)
        return {'acc': acc(logits, y)}

class LogisticRegression(Trainer):
    hparams = {'learning_rate': lambda: 10**np.random.uniform(-7, -3)}
    
    def train_step(self, batch):
        x, (y, y_e) = batch
        x, y, y_e = x.to(self.device), y.to(self.device), y_e.to(self.device)
        logits = self.classifier(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': val(loss), 'acc': acc(logits, y)}

class SVM(Trainer):
    hparams = {'learning_rate': lambda: 10**np.random.uniform(-7, -3),
               'weight_decay': lambda: 10**np.random.uniform(-6, -2)}
    
    def train_step(self, batch):
        x, (y, y_e) = batch
        x, y, y_e = x.to(self.device), y.to(self.device), y_e.to(self.device)
        logits = self.classifier(x)
        loss = nn.functional.multi_margin_loss(logits, y) + self.hparams['weight_decay']*self.classifier.weight.norm(p=2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': val(loss), 'acc': acc(logits, y)}

class VREx(Trainer):
    hparams = {'learning_rate': lambda: 10**np.random.uniform(-7, -3),
               'penalty_weight': lambda: 10**np.random.uniform(-1, 5),
               'anneal_iters': lambda: 10**np.random.uniform(0, 4)}
    
    def train_step(self, batch):
        if not hasattr(self, 'num_steps'):
            self.num_steps = 0
        self.num_steps += 1
        penalty_weight = self.hparams['penalty_weight'] if self.num_steps >= self.hparams['anneal_iters'] else 1.0
        if self.num_steps == self.hparams['anneal_iters']:
            self.optimizer = optim.Adam(self.classifier.parameters(), lr=self.hparams['learning_rate'])
        x, (y, y_e) = batch
        x, y, y_e = x.to(self.device), y.to(self.device), y_e.to(self.device)
        logits = self.classifier(x)
        loss = nn.functional.cross_entropy(logits, y, reduction='none')
        mean_loss = loss.mean()
        invariance_penalty = 0.0
        for env_idx in torch.unique(y_e):
            env_loss = loss[y_e==env_idx].mean()
            invariance_penalty += (mean_loss - env_loss) ** 2
        invariance_penalty /= len(torch.unique(y_e))
        vrex_loss = mean_loss + penalty_weight*invariance_penalty
        self.optimizer.zero_grad()
        vrex_loss.backward()
        self.optimizer.step()
        return {'loss': val(vrex_loss), 'acc': acc(logits, y)}

class IRM(Trainer):
    hparams = {'learning_rate': lambda: 10**np.random.uniform(-7, -3),
               'penalty_weight': lambda: 10**np.random.uniform(-1, 5),
               'anneal_iters': lambda: 10**np.random.uniform(0, 4)}
    
    def train_step(self, batch):
        if not hasattr(self, 'num_steps'):
            self.num_steps = 0
        self.num_steps += 1
        penalty_weight = self.hparams['penalty_weight'] if self.num_steps >= self.hparams['anneal_iters'] else 1.0
        if self.num_steps == self.hparams['anneal_iters']:
            self.optimizer = optim.Adam(self.classifier.parameters(), lr=self.hparams['learning_rate'])
        x, (y, y_e) = batch
        x, y, y_e = x.to(self.device), y.to(self.device), y_e.to(self.device)
        logits = self.classifier(x)
        empirical_risk, invariance_penalty = 0.0, 0.0
        for env_idx in torch.unique(y_e):
            logits_env = logits[y_e==env_idx]
            y_env = y[y_e==env_idx]
            empirical_risk += nn.functional.cross_entropy(logits_env, y_env)
            logits_1, logits_2 = logits_env[::2], logits_env[1::2]
            y_1, y_2 = y_env[::2], y_env[1::2]
            n = np.max((len(y_1), len(y_2)))
            logits_1, logits_2, y_1, y_2 = logits_1[:n], logits_2[:n], y_1[:n], y_2[:n]
            scale = torch.tensor(1.0, device=logits.device, requires_grad=True)
            loss_1 = nn.functional.cross_entropy(logits_1*scale, y_1)
            loss_2 = nn.functional.cross_entropy(logits_2*scale, y_2)
            grad_1 = grad(loss_1, [scale], create_graph=True)[0]
            grad_2 = grad(loss_2, [scale], create_graph=True)[0]
            invariance_penalty += (grad_1 * grad_2).sum()
        empirical_risk /= len(torch.unique(y_e))
        invariance_penalty /= len(torch.unique(y_e))
        irm_loss = empirical_risk + penalty_weight*invariance_penalty
        self.optimizer.zero_grad()
        irm_loss.backward()
        self.optimizer.step()
        return {'loss': val(irm_loss), 'acc': acc(logits, y)}

def run_epoch(dataloaders, trainer):
    if len(dataloaders) == 3:
        train_dataloader, val_dataloader, test_dataloader = dataloaders
    elif len(dataloaders) == 2:
        train_dataloader, val_dataloader = dataloaders
        test_dataloader = None
    rv = {}
    for batch in train_dataloader:
        step_rv = trainer.train_step(batch)
        for key, item in step_rv.items():
            if not 'train_'+key in rv.keys():
                rv['train_'+key] = []
            rv['train_'+key].append(item)
    for batch in val_dataloader:
        step_rv = trainer.eval_step(batch)
        for key, item in step_rv.items():
            if not 'val_'+key in rv.keys():
                rv['val_'+key] = []
            rv['val_'+key].append(item)
    if test_dataloader is not None:
        for batch in test_dataloader:
            step_rv = trainer.eval_step(batch)
            for key, item in step_rv.items():
                if not 'test_'+key in rv.keys():
                    rv['test_'+key] = []
                rv['test_'+key].append(item)
    for key, item in rv.items():
        rv[key] = np.mean(item)
    return rv
    
def get_baseline_results(dataloaders, device, epochs_per_trial=100):
    print('Evaluating accuracy of logistic regression trained on the target domain.')
    train_dataset = dataloaders[0].dataset.dataset
    train_dataset, _ = torch.utils.data.random_split(train_dataset, [len(train_dataset)-len(train_dataset)//2, len(train_dataset)//2])
    test_dataset = dataloaders[2].dataset
    target_train_dataset, target_test_dataset = torch.utils.data.random_split(test_dataset, [len(test_dataset)-len(test_dataset)//2, len(test_dataset)//2])
    full_dataset = torch.utils.data.ConcatDataset((train_dataset, target_train_dataset))
    num_features = train_dataset.dataset.num_features
    num_classes = train_dataset.dataset.num_classes
    train_dataloader = torch.utils.data.DataLoader(full_dataset, shuffle=True, batch_size=32)
    test_dataloader = torch.utils.data.DataLoader(target_test_dataset, shuffle=False, batch_size=32)
    logistic_regression_trainer = LogisticRegression(num_features, num_classes, device, {'learning_rate': 2e-4})
    baseline_results = {}
    for epoch_idx in range(epochs_per_trial):
        epoch_rv = run_epoch((train_dataloader, test_dataloader), logistic_regression_trainer)
        for key, item in epoch_rv.items():
            if not key in baseline_results.keys():
                baseline_results[key] = []
            baseline_results[key].append(item)
    best_epoch_idx = np.argmax(baseline_results['val_acc'])
    print('Best results when training on the target domain:')
    print('\t'+', '.join(['{}: {}'.format(key, item[best_epoch_idx]) for key, item in baseline_results.items()]))
    return baseline_results
    
def random_search_hparams(trainer_class, dataloaders, device, n_trials=20, epochs_per_trial=100):
    print('Sweeping hyperparams for trainer class {}'.format(trainer_class))
    num_features = dataloaders[0].dataset.dataset.num_features
    num_classes = dataloaders[0].dataset.dataset.num_classes
    results = []
    
    # Randomly sweep different sets of hyperparameters
    for trial_idx in tqdm(range(n_trials)):
        hparams = {
            hparam_name: get_hparam_fn() for hparam_name, get_hparam_fn in trainer_class.hparams.items()
        }
        trainer = trainer_class(num_features, num_classes, device, hparams)
        trial_results = {}
        for epoch_idx in range(epochs_per_trial):
            epoch_rv = run_epoch(dataloaders, trainer)
            for key, item in epoch_rv.items():
                if not key in trial_results.keys():
                    trial_results[key] = []
                trial_results[key].append(item)
        best_epoch_idx = np.argmax(trial_results['val_acc'])
        for key, item in trial_results.items():
            trial_results[key] = item[best_epoch_idx]
        results.append((hparams, trial_results))
    #best_trial_results, best_hparams = {'val_acc': -np.inf}, None
    best_holdout_results, best_holdout_hparams = {'val_acc': -np.inf}, None
    best_oracle_results, best_oracle_hparams = {'test_acc': -np.inf}, None
    for hparams, trial_results in results:
        if trial_results['val_acc'] > best_holdout_results['val_acc']:
            best_holdout_results = trial_results
            best_holdout_hparams = hparams
        if trial_results['test_acc'] > best_oracle_results['test_acc']:
            best_oracle_results = trial_results
            best_oracle_hparams = hparams
    print('Sweep complete.')
    print('\tBest holdout results: {}'.format(best_holdout_results))
    print('\tBest holdout hparams: {}'.format(best_holdout_hparams))
    print('\tBest oracle results: {}'.format(best_oracle_results))
    print('\tBest oracle hparams: {}'.format(best_oracle_hparams))
    print('\n\n')
    return {'best_holdout_results': best_holdout_results, 'best_holdout_hparams': best_holdout_hparams,
            'best_oracle_results': best_oracle_results, 'best_oracle_hparams': best_oracle_hparams,
            'all_results': results}

TRAINER_CLASSES = [
    LogisticRegression,
    SVM,
    VREx,
    IRM
]

def evaluate_trained_models(
    classifiers='all', seeds='all', datasets='all',
    overwrite=False, batch_size=32, device=None, num_epochs=100
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if 'all' in classifiers:
        classifiers = [c.__name__ for c in TRAINER_CLASSES]
    base_dir = os.path.join('.', 'results')
    for dataset in os.listdir(base_dir):
        if not('all' in datasets) and not(dataset in datasets):
            continue
        dataset_constructor = getattr(domainbed, dataset)
        for holdout_dir in os.listdir(os.path.join(base_dir, dataset)):
            holdout_domain = holdout_dir.split('_')[-1]
            for fe_type in ['random', 'imagenet_pretrained'] + list(os.listdir(os.path.join(base_dir, dataset, holdout_dir))) :
                if 'mixup' in fe_type:
                    continue
                for seed in seeds:
                    trial_dir = os.path.join(base_dir, dataset, holdout_dir, fe_type, 'trial_%d'%(seed))
                    os.makedirs(trial_dir, exist_ok=True)
                    dataloaders = get_dataloaders(dataset_constructor, holdout_domain, fe_type, seed,
                                                  batch_size=batch_size, device=device)
                    print('Finished generating dataloaders for {} / {} / {} / {}'.format(dataset, holdout_domain, fe_type, seed))
                    results_dir = os.path.join(base_dir, dataset, holdout_dir, fe_type, trial_dir, 'results', 'linear_classifiers')
                    os.makedirs(results_dir, exist_ok=True)
                    if overwrite or not(os.path.exists(os.path.join(results_dir, 'baseline_results.pickle'))):
                        baseline_results = get_baseline_results(dataloaders, device, num_epochs)
                        with open(os.path.join(results_dir, 'baseline_results.pickle'), 'wb') as F:
                            pickle.dump(baseline_results, F)
                    else:
                        print('Preexisting target-domain results exist; skipping baseline trial.')
                    for trainer_class in TRAINER_CLASSES:
                        if not trainer_class.__name__ in classifiers:
                            continue
                        if not(overwrite) and os.path.exists(os.path.join(results_dir, classifier_name+'.pickle')):
                            print('Found a pre-existing linear classifier for {} / {} / {} / {}'.format(
                                dataset, holdout_domain, fe_type, trainer_class))
                            continue
                        rv = random_search_hparams(trainer_class, dataloaders, device)
                        with open(os.path.join(results_dir, trainer_class.__name__+'.pickle'), 'wb') as F:
                            pickle.dump(rv, F)