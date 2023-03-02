# -*- coding: utf-8 -*-
"""
Created on Fri May 13 03:42:40 2022

@author: Bram
"""
import torch
import torch.nn as nn
from torchvision.models import vgg19_bn

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        
        self.vgg = vgg19_bn(pretrained=True)
        
        self.vgg.features = self.vgg.features
        
        
        # Modify the last layer
        number_features = self.vgg.classifier[6].in_features
        features = list(self.vgg.classifier.children())[:-1] # Remove last layer
        features.extend([torch.nn.Linear(number_features, 100)])
        self.vgg.classifier = torch.nn.Sequential(*features)
        
        # Freeze all layer
        for name, param in self.vgg.named_parameters():        
            param.requires_grad = False
                
        # for mod in list(self.vgg.features.modules()):
        #     if hasattr(mod, "inplace"):
        #         # print(mod)
        #         mod.inplace=False
        # for mod in list(self.vgg.classifier.modules()):
        #     if hasattr(mod, "inplace"):
        #         # print(mod)
        #         mod.inplace=False
            
        # Unfreeze last layer
        list(self.vgg.parameters())[-1].requires_grad = True
        list(self.vgg.parameters())[-2].requires_grad = True

    def forward(self, x):
        x = self.vgg(x)
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
        vgg = vgg19_bn(pretrained=True)
        features = list(vgg.features.children())
        features.extend([torch.nn.Conv2d(512, out_channels=100, kernel_size=3), torch.nn.AdaptiveAvgPool2d((1, 1))])
        self.vgg_FC = torch.nn.Sequential(*features)
        
        for idx, param in enumerate(self.vgg_FC.parameters()):
            if(idx < 50):
                param.requires_grad = False
        
    def forward(self, x):
        x = self.vgg_FC(x)
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
            if(key.find("features_conv") != -1):
                del state_dict[key]
        self.vgg_FC.load_state_dict(state_dict)