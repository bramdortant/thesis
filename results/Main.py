# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 18:06:51 2024

@author: Bram
"""

# Imports
import os
import torch

from torch.utils.data import DataLoader
from torchvision import transforms as T
 
from MiniImageNetResized import MiniImageNetResized
from Models import VGG19, VGG19_FC
from Trainer import Trainer
from Researcher import Researcher

import matplotlib.pyplot as plt
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PATH_vgg = "vgg_mini_imagenet.pth"
PATH_CAM = "vgg_FC_mini_imagenet.pth"

lr = 1e-4
vgg_num_epochs = 60
vgg_finetune_epochs = 40
CAM_num_epochs = 60
CAM_finetune_epochs = 40
batch_size = 16
cuda = False

num_workers = 2


if __name__ == "__main__":

    # Set up the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")
    if cuda:
        torch.backends.cudnn.benchmark = True

    # Setting up the data
    _data = MiniImageNetResized()

    _train, _test = torch.utils.data.random_split(_data, [int(len(_data)*0.8), int(len(_data)*0.2)])
    _train, _val = torch.utils.data.random_split(_train, [int(len(_train)*0.8), int(len(_train)*0.2)])

    # Setting up the data loaders
    _train_load = DataLoader(_train,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=True,
                             drop_last=True)
    _val_load = DataLoader(_val,
                           batch_size=batch_size,
                           num_workers=num_workers,
                           shuffle=False)
    _test_load = DataLoader(_test,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=False)

    # Setting up the standard transformations
    transform = T.Compose([
        T.Resize(224),
        T.Normalize(mean=[0.0, 0.0, 0.0], std=[1./255, 1./255, 1./255]),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Setting up the transformations for training with image augmentation
    augment_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.Normalize(mean=[0.0, 0.0, 0.0], std=[1./255, 1./255, 1./255]),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomRotation(degrees=(-20, 20)),
        T.RandomHorizontalFlip(p=0.5),
        T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2))
    ])

    # Load the model based on VGG19
    model = VGG19().to(device)
    # print(model)
    model_FC = VGG19_FC().to(device)
    # print(model_FC)

# Train________________________________________________________________________________________________________________________________________
    print("Train")
    if (os.path.exists(PATH_vgg)):
        # Load model
        model.set_weights(PATH_vgg)
        model.to(device)
    else:
        trainer = Trainer(device, model, _train_load, _val_load, transform, augment_transform, lr, PATH_vgg)
        trainer.train(vgg_num_epochs, vgg_finetune_epochs)
        trainer.test(_test_load)

    if (os.path.exists(PATH_CAM)):
        # Load model
        model_FC.set_weights(PATH_CAM)
        model_FC.to(device)
    else:
        trainer_FC = Trainer(device, model_FC, _train_load, _val_load, transform, augment_transform, lr, PATH_CAM)
        trainer_FC.train(CAM_num_epochs, CAM_finetune_epochs)
        trainer_FC.test(_test_load)

# Visualize_________________________________________________________________________________________________________________________________________________
    print("Visualize")
    
    # Unfreeze the layers
    for param in model.parameters():
        param.requires_grad = True

    # Unfreeze the layers
    for param in model_FC.parameters():
        param.requires_grad = True
    
    researcher = Researcher(device, model, model_FC, _train_load, _val_load, _test_load, transform)
    
    
    

    model.eval()
    model_FC.eval()

    img, lbl = _data[2000]
    # print(img.shape)
    # plt.figure()
    # plt.imshow(np.transpose(img.squeeze(), (1, 2, 0)))
    # plt.show()

    normed_img = transform(img.to(torch.float32)).unsqueeze(0)
    normed_img.requires_grad = True

    # pred = model(normed_img)
    # pred_FC = model_FC(normed_img)
    # # print(torch.argmax(pred))
    # # print(torch.argmax(pred_FC))
    # # print(torch.argmax(lbl))

    # img = np.transpose(T.Resize((224, 224))(img.squeeze()), (1, 2, 0))

    heatmap, CAM_mask = Researcher.CAM(researcher, model_FC, normed_img, lbl)
    # heatmapPP, maskPP = Researcher.Grad_CAMPP(researcher, model, normed_img, lbl)
    # integrated_gradients, mask_ig = Researcher.integrated_gradients(researcher, model, normed_img, lbl, 20)
    # guided_backprop, mask_gb, guided_gradcam, mask_gg = Researcher.guided_gradcam(researcher, model, normed_img, lbl, 30)
    
    # plt.figure()
    # plt.subplot(5, 4, 1)
    # plt.imshow(img)
    # plt.subplot(5, 4, 2)
    # plt.imshow(heatmap.detach())
    # plt.subplot(5, 4, 3)
    # plt.imshow(CAM_mask)
    # plt.subplot(5, 4, 4)
    # plt.imshow(img.detach())
    # plt.imshow(heatmap.detach(), alpha=0.7)
    # plt.subplot(5, 4, 5)
    # plt.imshow(img)
    # plt.subplot(5, 4, 6)
    # plt.imshow(heatmapPP)
    # plt.subplot(5, 4, 7)
    # plt.imshow(maskPP)
    # plt.subplot(5, 4, 8)
    # plt.imshow(img.detach())
    # plt.imshow(heatmapPP.detach(), alpha=0.7)
    # plt.subplot(5, 4, 9)
    # plt.imshow(img)
    # plt.subplot(5, 4, 10)
    # plt.imshow(integrated_gradients)
    # plt.subplot(5, 4, 11)
    # plt.imshow(mask_ig)
    # plt.subplot(5, 4, 12)
    # plt.imshow(img)
    # plt.imshow(integrated_gradients, alpha=0.7)
    # plt.subplot(5, 4, 13)
    # plt.imshow(img.detach())
    # plt.subplot(5, 4, 14)
    # plt.imshow(guided_backprop.detach())
    # plt.subplot(5, 4, 15)
    # plt.imshow(mask_gb)
    # plt.subplot(5, 4, 16)
    # plt.imshow(img.detach())
    # plt.imshow(guided_backprop, alpha = 0.7)
    # plt.subplot(5, 4, 17)
    # plt.imshow(img)
    # plt.subplot(5, 4, 18)
    # plt.imshow(guided_gradcam)
    # plt.subplot(5, 4, 19)
    # plt.imshow(mask_gg.squeeze())
    # plt.subplot(5, 4, 20)
    # plt.imshow(img.detach())
    # plt.imshow(guided_gradcam, alpha = 0.7)
    # plt.show()
    
    # print(torch.max(heatmap))
    # print(torch.min(heatmap))
    # print(torch.max(heatmapPP))
    # print(torch.min(heatmapPP))
    # print(torch.max(integrated_gradients))
    # print(torch.min(integrated_gradients))
    # print(torch.max(guided_backprop))
    # print(torch.min(guided_backprop))
    # print(torch.max(guided_gradcam))
    # print(torch.min(guided_gradcam))
    
    # print("hoi")
    
    # print(torch.max(CAM_mask))
    # print(torch.min(CAM_mask))
    # print(torch.max(maskPP))
    # print(torch.min(maskPP))
    # print(torch.max(mask_ig))
    # print(torch.min(mask_ig))
    # print(torch.max(mask_gb))
    # print(torch.min(mask_gb))
    # print(torch.max(mask_gg))
    # print(torch.min(mask_gg))
    
    
    # plt.figure()
    # plt.subplot(5, 1, 1)
    # plt.imshow(Researcher.mask(img, CAM_mask, 0.5))
    # plt.subplot(5, 1, 2)
    # plt.imshow(Researcher.mask(img, maskPP, 0.5))
    # plt.subplot(5, 1, 3)
    # plt.imshow(Researcher.mask(img, mask_ig, 0.5))
    # plt.subplot(5, 1, 4)
    # plt.imshow(Researcher.mask(img, mask_gb, 0.5))
    # plt.subplot(5, 1, 5)
    # plt.imshow(Researcher.mask(img, mask_gg, 0.5))
    # plt.show()

    # plt.figure()
    # plt.subplot(7, 3, 1)
    # plt.imshow(integrated_gradients1)
    # plt.subplot(7, 3, 2)
    # plt.imshow(mask_ig1)
    # plt.subplot(7, 3, 3)
    # plt.imshow(Researcher.mask(img, mask_ig1, 0.5))
    # plt.subplot(7, 3, 4)
    # plt.imshow(integrated_gradients2)
    # plt.subplot(7, 3, 5)
    # plt.imshow(mask_ig2)
    # plt.subplot(7, 3, 6)
    # plt.imshow(Researcher.mask(img, mask_ig2, 0.5))
    # plt.subplot(7, 3, 7)
    # plt.imshow(integrated_gradients3)
    # plt.subplot(7, 3, 8)
    # plt.imshow(mask_ig3)
    # plt.subplot(7, 3, 9)
    # plt.imshow(Researcher.mask(img, mask_ig3, 0.5))
    # plt.subplot(7, 3, 10)
    # plt.imshow(integrated_gradients4)
    # plt.subplot(7, 3, 11)
    # plt.imshow(mask_ig4)
    # plt.subplot(7, 3, 12)
    # plt.imshow(Researcher.mask(img, mask_ig4, 0.3))
    # plt.subplot(7, 3, 13)
    # plt.imshow(integrated_gradients2)
    # plt.subplot(7, 3, 14)
    # plt.imshow(mask_ig2)
    # plt.subplot(7, 3, 15)
    # plt.imshow(Researcher.mask(img, mask_ig2, 0.5))
    # plt.subplot(7, 3, 16)
    # plt.imshow(integrated_gradients2)
    # plt.subplot(7, 3, 17)
    # plt.imshow(mask_ig2)
    # plt.subplot(7, 3, 18)
    # plt.imshow(Researcher.mask(img, mask_ig2, 0.7))
    # plt.subplot(7, 3, 19)
    # plt.imshow(integrated_gradients2)
    # plt.subplot(7, 3, 20)
    # plt.imshow(mask_ig2)
    # plt.subplot(7, 3, 21)
    # plt.imshow(Researcher.mask(img, mask_ig2, 0.9))
    # plt.show()


# Research_____________________________________________________________________________________________________________________________________
    print("Research")
    # researcher = Researcher(device, model, model_FC, _train_load, _val_load, _test_load, transform)
    results = researcher.research()

    print(results)
