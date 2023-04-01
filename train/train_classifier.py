import torch
from torch import nn, optim
from extracted_features_dataset import ExtractedFeaturesDataset

# Types of linear classifiers to implement:
#   ERM (logistic regression, SVM, reweighted by dataset size)
#   Invariant risk minimization
#   Group distributionally-robust optimization
#   Domain adjusted regression

def train_classifier_erm(
    model=None, optimizer=None, train_dataloader=None, val_dataloader=None, downstream_dataloader=None,
    device=None, objective=None):
    assert all(arg is not None for arg in locals())
    assert objective in ['logistic', 'svm', 'logistic_reweighted']
    if objecive == 'logistic':
        loss_fn = nn.CrossEntropyLoss()

def construct_linear_classifier(
    dataset_constructor=None,
    train_dataset_kwargs=None,
    downstream_dataset_kwargs=None,
    dataloader_kwargs=None,
    feature_extractor=None,
    optimizer_constructor=None,
    optimizer_kwargs=None,
    device=None):
    assert all(arg is not None for arg in locals())
    model = nn.Linear(feature_extractor.features, dataset.num_classes).to(device)
    optimizer = optimizer_constructor(model.parameters(), **optimizer_kwargs)
    raw_train_dataset = dataset_constructor(**train_dataset_kwargs)
    raw_downstream_dataset = dataset_constructor(**downstream_dataset_kwargs)
    train_dataset = ExtractedFeaturesDataset(
        raw_train_dataset, feature_extractor,
        batch_size=1 if not('batch_size' in dataloader_kwargs.keys()) else dataloader_kwargs['batch_size'],
        device=device
    )
    downstream_dataset = ExtractedFeaturesDataset(
        raw_downstream_dataset, feature_extractor,
        batch_size=1 if not('batch_size' in dataloader_kwargs.keys()) else dataloader_kwargs['batch_size'],
        device=device
    )
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [4*len(train_dataset)//5, len(train_dataset)//5])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)
    downstream_dataloader = torch.utils.data.DataLoader(downstream_dataset, shuffle=False, **dataloader_kwargs)
    return {'model': model, 'optimizer': optimizer, 'train_dataloader': train_dataloader, 'val_dataloader': val_dataloader,
            'downstream_dataloader': downstream_dataloader, 'device': device}