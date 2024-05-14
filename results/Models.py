# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 02:52:18 2024

@author: Bram
"""

import torch
import torch.nn as nn
from torchvision.models import vgg19_bn

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        
        self.vgg = vgg19_bn(weights='DEFAULT')
        
        # Modify the last layer
        number_features = self.vgg.classifier[6].in_features
        features = list(self.vgg.classifier.children())[:-1] # Remove last layer
        features.extend([torch.nn.Linear(number_features, 100)])
        self.vgg.classifier = torch.nn.Sequential(*features)
        
        # Freeze all layer
        for param in self.vgg.parameters():        
            param.requires_grad = False
            
        # Make all ReLU's inplace for Captum functions to work
        for i, (name, layer) in enumerate(self.vgg.features.named_modules()):
            if isinstance(layer, nn.ReLU):
                self.vgg.features[int(name)] = nn.ReLU(inplace=False)
        for i, (name, layer) in enumerate(self.vgg.classifier.named_modules()):
            if isinstance(layer, nn.ReLU):
                self.vgg.classifier[int(name)] = nn.ReLU(inplace=False)
            
        # Unfreeze last layer
        list(self.vgg.parameters())[-1].requires_grad = True
        list(self.vgg.parameters())[-2].requires_grad = True

    def forward(self, x):
        # x = self.vgg(x)
        x = self.vgg.features(x)
        x = torch.flatten(x, 1)
        x = self.vgg.classifier(x)
        return x
    
    # set weights
    def set_weights(self, PATH):
        state_dict = None
        if torch.cuda.is_available():
            state_dict = torch.load(PATH, map_location="cuda:0")
        else:
            state_dict = torch.load(PATH, map_location=torch.device('cpu'))
        
        for key in list(state_dict.keys()):
            state_dict[key.replace('vgg.', '')] = state_dict.pop(key)
        self.vgg.load_state_dict(state_dict)
        
        
class VGG19_FC(nn.Module):
    def __init__(self):
        super(VGG19_FC, self).__init__()
        self.vgg_FC = vgg19_bn(weights='DEFAULT')
        features = list(self.vgg_FC.features.children())
        features.extend([torch.nn.Conv2d(512, out_channels=100, kernel_size=3), torch.nn.AdaptiveAvgPool2d((1, 1))])
        self.vgg_FC.features = torch.nn.Sequential(*features)
        
        self.vgg_FC.classifier = torch.nn.Linear(100, 100)
        
        for param in self.vgg_FC.parameters():
            param.requires_grad = False
            
        list(self.vgg_FC.parameters())[-1].requires_grad = True
        list(self.vgg_FC.parameters())[-2].requires_grad = True
        list(self.vgg_FC.parameters())[-3].requires_grad = True
        list(self.vgg_FC.parameters())[-4].requires_grad = True
        
    def forward(self, x):
        x = self.vgg_FC.features(x)
        x = torch.flatten(x, 1)
        x = self.vgg_FC.classifier(x)
        return x
    
    # set weights
    def set_weights(self, PATH):
        state_dict = None
        if torch.cuda.is_available():
            state_dict = torch.load(PATH, map_location="cuda:0")
        else:
            state_dict = torch.load(PATH, map_location=torch.device('cpu'))
        
        for key in list(state_dict.keys()):
            state_dict[key.replace('vgg_FC.', '')] = state_dict.pop(key)
            
        self.vgg_FC.load_state_dict(state_dict)
