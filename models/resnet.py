import numpy as np
import torch
import torchvision
from torch import nn, optim
from torch.nn.utils import spectral_norm
from models.common import *

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 use_spectral_norm=False,
                 use_batch_norm=False,
                 downsample=False,
                 upsample=False,
                 activation=lambda: nn.ReLU(inplace=True)):
        super().__init__()
        
        residual_modules = []
        if use_batch_norm:
            residual_modules.append(nn.BatchNorm2d(in_channels))
        residual_modules.append(activation())
        residual_modules.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        if use_spectral_norm:
            residual_modules[-1] = spectral_norm(residual_modules[-1])
        if use_batch_norm:
            residual_modules.append(nn.BatchNorm2d(out_channels))
        residual_modules.append(activation())
        if downsample:
            residual_modules.append(nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0))
        elif upsample:
            residual_modules.append(nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0))
        else:
            residual_modules.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        if use_spectral_norm:
            residual_modules[-1] = spectral_norm(residual_modules[-1])
        self.residual_connection = nn.Sequential(*residual_modules)
        
        skip_modules = []
        if downsample:
            skip_modules.append(nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0))
        elif upsample:
            skip_modules.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0))
        elif in_channels != out_channels:
            skip_modules.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0))
        if use_spectral_norm and len(skip_modules)>0:
            skip_modules[-1] = spectral_norm(skip_modules[-1])
        self.skip_connection = nn.Sequential(*skip_modules)
        
    def forward(self, x):
        x_rc = self.residual_connection(x)
        x_sc = self.skip_connection(x)
        out = x_rc + x_sc
        return out

class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, downsample_blocks, endomorphic_blocks,
                 use_spectral_norm=False,
                 use_batch_norm=False,
                 use_dropout=False,
                 pooling_fn=torch.mean,
                 activation=lambda: nn.ReLU(inplace=True)):
        super().__init__()
        
        self.features = out_channels
        self.input_transform = nn.Conv2d(in_channels, out_channels//(2**downsample_blocks), kernel_size=3, stride=1, padding=1)
        if use_spectral_norm:
            self.input_transform = spectral_norm(self.input_transform)
        fe_modules = []
        for n in range(downsample_blocks):
            fe_modules.append(ResidualBlock(
                out_channels//(2**(downsample_blocks-n)), out_channels//(2**(downsample_blocks-n-1)), downsample=True,
                use_spectral_norm=use_spectral_norm, use_batch_norm=use_batch_norm, activation=activation
            ))
        for n in range(endomorphic_blocks):
            fe_modules.append(ResidualBlock(
                out_channels, out_channels,
                use_spectral_norm=use_spectral_norm, use_batch_norm=use_batch_norm, activation=activation
            ))
        self.feature_extractor = nn.Sequential(*fe_modules)
        pooling_modules = [GlobalPool2d(pooling_fn)]
        if use_dropout:
            pooling_modules.append(nn.Dropout(p=0.5))
        self.pooling_layer = nn.Sequential(*pooling_modules)
        
    def get_features(self, x):
        x_i = self.input_transform(x)
        out = self.feature_extractor(x_i)
        return out
        
    def forward(self, x):
        x_i = self.input_transform(x)
        x_fe = self.feature_extractor(x_i)
        out = self.pooling_layer(x_fe)
        return out

class FeatureReconstructor(nn.Module):
    def __init__(self, in_channels, out_channels, output_shape, upsample_blocks, endomorphic_blocks,
                 use_spectral_norm=False,
                 use_batch_norm=False,
                 activation=lambda: nn.ReLU(inplace=True)):
        super().__init__()
        
        #self.input_transform = nn.ConvTranspose2d(
        #    in_channels, out_channels*2**upsample_blocks,
        #    kernel_size=output_shape[1]//(2**upsample_blocks), stride=1, padding=0)
        self.input_transform = nn.Conv2d(
            in_channels, out_channels*2**upsample_blocks,
            kernel_size=3, stride=1, padding=1)
        if use_spectral_norm:
            self.input_transform = spectral_norm(self.input_transform)
        modules = []
        for n in range(upsample_blocks):
            modules.append(ResidualBlock(
                out_channels*2**(upsample_blocks-n), out_channels*2**(upsample_blocks-n-1), upsample=True,
                use_spectral_norm=use_spectral_norm, use_batch_norm=use_batch_norm, activation=activation
            ))
        for n in range(endomorphic_blocks-1):
            modules.append(ResidualBlock(
                out_channels, out_channels,
                use_spectral_norm=use_spectral_norm, use_batch_norm=use_batch_norm, activation=activation
            ))
        modules.append(ResidualBlock(
            out_channels, output_shape[0],
            use_spectral_norm=use_spectral_norm, use_batch_norm=use_batch_norm, activation=activation
            
        ))
        self.model = nn.Sequential(*modules)
    
    def forward(self, x):
        x_i = self.input_transform(x)
        out = torch.tanh(self.model(x_i))
        return out

class Classifier(nn.Module):
    def __init__(self, input_shape, output_classes, features=64, resample_blocks=2, endomorphic_blocks=2, use_dropout=False):
        super().__init__()
        
        self.feature_extractor = FeatureExtractor(
            input_shape[0], features, resample_blocks, endomorphic_blocks, use_dropout=use_dropout, pooling_fn=torch.mean,
            use_spectral_norm=False, use_batch_norm=True, activation=lambda: nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(features, output_classes)
        
    def get_features(self, x):
        return self.feature_extractor(x)
    
    def classify_features(self, x):
        return self.classifier(x)

class Discriminator(nn.Module):
    def __init__(self, input_shape, output_classes, features=64, resample_blocks=2, endomorphic_blocks=2):
        super().__init__()
        
        self.feature_extractor = FeatureExtractor(
            input_shape[0], features, resample_blocks, endomorphic_blocks, pooling_fn=torch.sum, 
            use_spectral_norm=True, use_batch_norm=False, activation=lambda: nn.LeakyReLU(0.1)
        )
        self.realism_classifier = spectral_norm(nn.Linear(features, 1))
        self.label_classifier = spectral_norm(nn.Linear(features, output_classes))
        
    def get_features(self, x):
        return self.feature_extractor(x)
    
    def classify_realism(self, x):
        return self.realism_classifier(x)
    
    def classify_label(self, x):
        return self.label_classifier(x)

class Generator(nn.Module):
    def __init__(self, output_shape, num_classes, features=64, resample_blocks=2, endomorphic_blocks=2):
        super().__init__()
        
        self.latent_features = features
        self.input_transform = nn.ConvTranspose2d(
            features, features*2**resample_blocks,
            kernel_size=output_shape[1]//(2**resample_blocks), stride=1, padding=0)
        self.feature_reconstructor = FeatureReconstructor(
            features, features//4, output_shape, resample_blocks, endomorphic_blocks,
            use_spectral_norm=True, use_batch_norm=True, activation=lambda: nn.ReLU(0.1)
        )
        
    def forward(self, x):
        x_i = self.input_transform(x.view(-1, self.latent_features, 1, 1))
        return self.feature_reconstructor(x_i)

class Autoencoder(nn.Module):
    def __init__(self, input_shape, output_classes, features=64, resample_blocks=2, endomorphic_blocks=2,
                 use_spectral_norm=True):
        super().__init__()
        
        self.features = features
        self.feature_extractor = FeatureExtractor(
            input_shape[0], features, resample_blocks, endomorphic_blocks, pooling_fn=torch.mean, 
            use_spectral_norm=use_spectral_norm, use_batch_norm=True, activation=lambda: nn.ReLU(inplace=True)
        )
        self.feature_reconstructor = FeatureReconstructor(
            features, features//4, input_shape, resample_blocks, 0,
            use_spectral_norm=use_spectral_norm, use_batch_norm=True, activation=lambda: nn.ReLU(inplace=True)
        )
        self.label_classifier = nn.Linear(features, output_classes)
        if use_spectral_norm:
            self.label_classifier = spectral_norm(self.label_classifier)
        
    def get_features(self, x):
        return self.feature_extractor.get_features(x)
    
    def reconstruct_features(self, x):
        return self.feature_reconstructor(x)
    
    def classify_labels(self, x):
        return self.label_classifier(self.feature_extractor.pooling_layer(x))
    
    def forward(self, x):
        return self.reconstruct_features(self.get_features(x))

    
# Based off of DomainBed implementation here:
#   https://github.com/facebookresearch/DomainBed/blob/main/domainbed/networks.py
class PretrainedRN50(nn.Module):
    def __init__(self, input_shape, n_outputs, *args, pretrained=False, **kwargs):
        super().__init__()
        
        self.num_features = 2048
        self.feature_extractor = torchvision.models.resnet50(pretrained=pretrained)
        del self.feature_extractor.fc
        self.feature_extractor.fc = nn.Identity()
        self.classifier = nn.Linear(self.num_features, n_outputs)
        self.freeze_bn()
        
    def get_features(self, x):
        x_fe = self.feature_extractor(x)
        x_fe = x_fe.view(-1, x_fe.size(1))
        return x_fe
    
    def classify_features(self, x):
        return self.classifier(x)
    
    def forward(self, x):
        return self.classify_features(self.get_features(x))
    
    def train(self, mode=True):
        super().train(mode)
        self.freeze_bn()
        
    def freeze_bn(self):
        for m in self.feature_extractor.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()