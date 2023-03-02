# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 13:49:02 2022

@author: Bram
"""

# Imports
import os
import argparse
import numpy as np
import torch

from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms as T
# from torchvision.utils import make_grid, save_image


from MiniImagenetResized import MiniImagenetResized
from Models import VGG19, VGG19_FC

from GradCamPP import GradCAM, GradCAMpp, visualize_cam
from captum.attr import IntegratedGradients, GuidedGradCam


import matplotlib.pyplot as plt

PATH_vgg = "vgg_mini_imagenet.pth"
#"vgg_no_inline.pth"#
PATH_CAM = "vgg_FC_no_inline.pth"#"vgg_FC_mini_imagenet.pth"

# Initialize argument parser for easy
parser = argparse.ArgumentParser("MiniImagenet")

# Model arguments
parser.add_argument("--num_filters", type=int, default=16)

# Training arguments
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate hyperparameter for the AdamW optimizer")
# parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay hyperparameter for the AdamW optimizer")
parser.add_argument("--num_epochs", type=int, default=20, help="Number of runs over all of the training data")
parser.add_argument("--batch_size", type=int, default=2, help="The batchsize that is fed to the model")
parser.add_argument("--num_workers", type=int, default=1, help="The number of processes that work for the dataloader")
# parser.add_argument("--device", type=str, default="cuda:0", choices=["cuda:0", "cpu:0"], help="Specify the device on which you run the code, gpu(cuda) or cpu")
parser.add_argument("--benchmark_cudnn", type=bool, default=False)

if __name__ == "__main__":
    #Load the model based on VGG19
    vgg = VGG19()
    # print(vgg)
    vgg_FC = VGG19_FC()
    # print(vgg_FC)
    
    # Set up the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")
        
    # Load the weights
    if(os.path.exists(PATH_vgg)):
        # Load model
        vgg.set_weights(PATH_vgg)
        vgg.to(device)
    if(os.path.exists(PATH_CAM)):
        # Load model
        vgg_FC.set_weights(PATH_CAM)
        vgg_FC.to(device)
    
    # Setting up the data
    _data = MiniImagenetResized()
    
    # _train, _test = torch.utils.data.random_split(_data, [int(len(_data)*0.8), int(len(_data)*0.2)])
    # _train, _val = torch.utils.data.random_split(_train, [int(len(_train)*0.8), int(len(_train)*0.2)])
    # _train_load = DataLoader(_train,
    #                     batch_size=1,
    #                     num_workers=1,
    #                     shuffle=True,
    #                     drop_last=True)   
    # _val_load = DataLoader(_val,
    #                     batch_size=1,
    #                     num_workers=1,
    #                     shuffle=False)   
    # _test_load = DataLoader(_test,
                        # batch_size=1,
                        # num_workers=1,
                        # shuffle=False)
    
    # Setting up the transformations
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.Normalize(mean=[0.0, 0.0, 0.0], std=[1./255, 1./255, 1./255]),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    
    img, lbl = _data[0]
    print(type(img))
    img = img[None, :]
    print(type(img))
    normed_img = transform(img.float())
    pred = vgg(normed_img)
    pred_FC = vgg_FC(normed_img)
    # img.requires_grad = True
    
    # Unfreeze the layers
    for param in vgg.parameters():
        param.requires_grad = True
    
#CAM_________________________________________________________________________________________
    
    #     #TODO: implement CAM
    #     # https://tree.rocks/get-heatmap-from-cnn-convolution-neural-network-aka-grad-cam-222e08f57a34
    
    #     # conv_output = vgg_FC.get_layer("conv5_block3_out").output
    #     # pred_ouptut = res_model.get_layer("predictions").output
    #     # model = Model(res_model.input, outputs=[conv_ouptut, pred_layer])
    
    #     heatmapCAM = img
    
    
    #TODO: implement CAM
    # https://pytorch.org/hub/pytorch_vision_fcn_resnet101/
    # https://tree.rocks/get-heatmap-from-cnn-convolution-neural-network-aka-grad-cam-222e08f57a34
    
    
    # ______________________________
    
    # params = list(vgg_FC.parameters())
    # weight = np.squeeze(params[-1].data.numpy())
    
    # def return_CAM(feature_conv, weight, class_idx):
    #     # generate the class -activation maps upsample to 256x256
    #     size_upsample = (256, 256)
    #     bz, nc, h, w = feature_conv.shape
    #     output_cam = []
    #     for idx in class_idx:
    #         beforeDot =  feature_conv.reshape((nc, h*w))
    #         cam = np.matmul(weight[idx], beforeDot)
    #         cam = cam.reshape(h, w)
    #         cam = cam - np.min(cam)
    #         cam_img = cam / np.max(cam)
    #         cam_img = np.uint8(255 * cam_img)
    #         output_cam.append(cv2.resize(cam_img, size_upsample))
    #     return output_cam
    
    
    # # heatmapCAM = img
    
#Grad-CAM++_________________________________________________________________________________________
    
    gradcampp = GradCAMpp.from_config(model_type='vgg', arch=vgg, layer_name='features_50')
    # get a GradCAM saliency map on the class index 10.
    maskPP, logitPP = gradcampp(normed_img, class_idx=torch.argmax(F.softmax(pred[0], dim=0), dim= 0))
    # make heatmap from mask and synthesize saliency map using heatmap and img
    heatmapPP, camPP_result = visualize_cam(maskPP, normed_img)
    
    heatmapPP = torch.swapaxes(heatmapPP, 0, 2)
    camPP_result = torch.swapaxes(camPP_result, 0, 2)
    
    
#Integrated gradients_________________________________________________________________________________________
    
    vgg.eval()
    def attribute_image_features(algorithm, input, **kwargs):
        vgg.zero_grad()
        tensor_attributions = algorithm.attribute(input,
                                                  target=torch.argmax(pred[0], dim=0),
                                                  **kwargs
                                                  )
        
        return tensor_attributions
    
    ig = IntegratedGradients(vgg)
    attr_ig, delta = attribute_image_features(ig, normed_img, baselines=normed_img * 0, return_convergence_delta=True)
    attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))

    
#Guided Grad-CAM_________________________________________________________________________________________
    
    guided_gc = GuidedGradCam(vgg, vgg.vgg.features[49])
    attribution = guided_gc.attribute(normed_img, torch.argmax(F.softmax(pred[0], dim=0), dim= 0))
    
#Visualize_________________________________________________________________________________________
    # print(type(img))
    img = torch.swapaxes(img, 1, 3).squeeze()
    # heatmapCAM = torch.swapaxes(heatmapCAM, 1, 3).squeeze()
    heatmapPP = torch.swapaxes(heatmapPP, 0, 2)
    camPP_result = torch.swapaxes(camPP_result, 0, 2)
    # attr_ig = torch.swapaxes(attr_ig, 1, 3).squeeze()
    # attribution = torch.swapaxes(attribution, 1, 3).squeeze()
    print(torch.amax(heatmapPP))
    print(torch.amax(camPP_result))
 
    plt.figure()
    # plt.subplot(4, 3, 1)
    # plt.imshow(img)
    # plt.subplot(4, 3, 2)
    # plt.imshow(heatmapCAM.detach())
    # plt.subplot(4, 3, 3)
    # plt.imshow(img.detach())
    # plt.imshow(heatmapCAM.detach(), alpha = 0.7)
    plt.subplot(4, 3, 4)
    plt.imshow(img)
    plt.subplot(4, 3, 5)
    plt.imshow(heatmapPP)
    plt.subplot(4, 3, 6)
    plt.imshow(camPP_result)
    # plt.subplot(4, 3, 7)
    # plt.imshow(img)
    # plt.subplot(4, 3, 8)
    # plt.imshow(attr_ig)
    # plt.subplot(4, 3, 9)
    # plt.imshow(img)
    # plt.imshow(attr_ig, alpha = 0.7)
    # plt.subplot(4, 3, 10)
    # plt.imshow(img.detach())
    # plt.subplot(4, 3, 11)
    # plt.imshow(attribution.detach())
    # plt.subplot(4, 3, 12)
    # plt.imshow(img.detach())
    # plt.imshow(attribution.detach(), alpha = 0.7)
    plt.show()
