import numpy as np
import torch
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
            residual_modules.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1))
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
                 activation=lambda: nn.ReLU(inplace=True)):
        super().__init__()
        
        self.input_transform = nn.Conv2d(in_channels, out_channels//(2**downsample_blocks), kernel_size=3, stride=1, padding=1)
        if use_spectral_norm:
            self.input_transform = spectral_norm(self.input_transform)
        modules = []
        for n in range(downsample_blocks):
            modules.append(ResidualBlock(
                out_channels//(2**(downsample_blocks-n)), out_channels//(2**(downsample_blocks-n-1)), downsample=True,
                use_spectral_norm=use_spectral_norm, use_batch_norm=use_batch_norm, activation=activation
            ))
        for n in range(endomorphic_blocks):
            modules.append(ResidualBlock(
                out_channels, out_channels,
                use_spectral_norm=use_spectral_norm, use_batch_norm=use_batch_norm, activation=activation
            ))
        modules.append(GlobalPool2d(torch.sum))
        self.model = nn.Sequential(*modules)
        
    def forward(self, x):
        x_i = self.input_transform(x)
        out = self.model(x_i)
        return out

class FeatureReconstructor(nn.Module):
    def __init__(self, in_channels, out_channels, output_shape, upsample_blocks, endomorphic_blocks,
                 use_spectral_norm=False,
                 use_batch_norm=False,
                 activation=lambda: nn.ReLU(inplace=True)):
        super().__init__()
        
        self.input_transform = nn.ConvTranspose2d(
            in_channels, out_channels*2**upsample_blocks,
            kernel_size=output_shape[1]//(2**upsample_blocks), stride=1, padding=0)
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
        out = self.model(x_i)
        return out

class Classifier(nn.Module):
    def __init__(self, input_shape, output_classes=10, features=64, downsample_blocks=2, endomorphic_blocks=2):
        super().__init__()
        
        self.feature_extractor = FeatureExtractor(
            input_shape[0], features, downsample_blocks, endomorphic_blocks,
            use_spectral_norm=False, use_batch_norm=True, activation=lambda: nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(features, output_classes)
        
    def forward(self, x):
        x_fe = self.feature_extractor(x)
        out = self.classifier(x_fe)
        return out

class Discriminator(nn.Module):
    def __init__(self, input_shape, output_classes=10, features=64, downsample_blocks=2, endomorphic_blocks=2):
        super().__init__()
        
        self.feature_extractor = FeatureExtractor(
            input_shape[0], features, downsample_blocks, endomorphic_blocks,
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
    def __init__(self, output_shape, latent_features=64, upsample_blocks=2, endomorphic_blocks=2):
        super().__init__()
        
        self.latent_features = latent_features
        self.feature_reconstructor = FeatureReconstructor(
            latent_features, latent_features//4, output_shape, upsample_blocks, endomorphic_blocks,
            use_spectral_norm=True, use_batch_norm=True, activation=lambda: nn.ReLU(0.1)
        )
        
    def forward(self, x):
        return self.feature_reconstructor(x.view(-1, self.latent_features, 1, 1))

class Autoencoder(nn.Module):
    def __init__(self, input_shape, output_classes=10, bottleneck_features=64, resample_blocks=2, endomorphic_blocks=2,
                 use_spectral_norm=True):
        super().__init__()
        
        self.bottleneck_features = bottleneck_features
        self.feature_extractor = FeatureExtractor(
            input_shape[0], bottleneck_features, resample_blocks, endomorphic_blocks,
            use_spectral_norm=use_spectral_norm, use_batch_norm=True, activation=lambda: nn.ReLU(inplace=True)
        )
        self.feature_reconstructor = FeatureReconstructor(
            bottleneck_features, bottleneck_features//4, input_shape, resample_blocks, 0,
            use_spectral_norm=use_spectral_norm, use_batch_norm=True, activation=lambda: nn.ReLU(inplace=True)
        )
        self.label_classifier = nn.Linear(bottleneck_features, output_classes)
        if use_spectral_norm:
            self.label_classifier = spectral_norm(self.label_classifier)
        
    def get_features(self, x):
        return self.feature_extractor(x)
    
    def reconstruct_features(self, x):
        return self.feature_reconstructor(x.view(-1, self.bottleneck_features, 1, 1))
    
    def classify_labels(self, x):
        return self.label_classifier(x)