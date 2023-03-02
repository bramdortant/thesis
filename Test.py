import argparse
import os
import torch

from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms as T
import matplotlib.pyplot as plt
# import numpy as np
from tqdm import tqdm
# from collections import OrderedDict

from AverageMeter import EpochAverageMeter, RunningAverageMeter
from MiniImagenetResized import MiniImagenetResized
from GradCamPP import GradCAM, GradCAMpp, visualize_cam
from Models import VGG19, VGG19_FC

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
    args = parser.parse_args()
    
    # Setting up the data
    _data = MiniImagenetResized()
    
    _train, _test = torch.utils.data.random_split(_data, [int(len(_data)*0.8), int(len(_data)*0.2)])
    _train, _val = torch.utils.data.random_split(_train, [int(len(_train)*0.8), int(len(_train)*0.2)])
    _train_load = DataLoader(_train,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        shuffle=True,
                        drop_last=True)
    _val_load = DataLoader(_val,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        shuffle=False)   
    _test_load = DataLoader(_test,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        shuffle=False)   
    
    # Setting up the transformations
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.Normalize(mean=[0.0, 0.0, 0.0], std=[1./255, 1./255, 1./255]),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Setting up metric trackers
    trn_loss = RunningAverageMeter()
    trn_accy = RunningAverageMeter()
    val_loss = EpochAverageMeter()
    val_accy = EpochAverageMeter()
    
    # Set up the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")
    if(args.benchmark_cudnn):
        torch.backends.cudnn.benchmark = True
    
    # Setting up the model
    model = VGG19()
    
    # Move model to device
    model = model.to(device)
    
    # Setting up optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    loss_func = torch.nn.CrossEntropyLoss()
    
    PATH = "vgg_mini_imagenet.pth"
    
    if(os.path.exists(PATH)):
        # Load model
        model.set_weights(PATH)
        model.to(device)
    else:
        # Training loop
        for epoch_nb in range(args.num_epochs):
            # Metrics
            trn_loss.reset()
            trn_accy.reset()
            
            # Training loop
            loop = tqdm(_train_load, position=0, leave=True)
            loop.set_description_str(f"Epoch: {epoch_nb}")
            for imgs, lbls in loop:
                imgs = transform(imgs.to(torch.float32).to(device))
                lbls = lbls.to(device)
                optimizer.zero_grad()
                
                # Forward pass
                out = model(imgs)
                lss = loss_func(out, lbls.max(dim=1)[1])
                
                # Backward pass
                lss.backward()
                optimizer.step()
                
                # Update metrics 
                trn_loss.update(lss.item())
                trn_accy.update(torch.mean((out.max(dim=1)[1] == lbls.max(dim=1)[1]).to(torch.float32)))
                
                loop.set_postfix_str(f"loss: {trn_loss.avg:.3f} - accuracy: {trn_accy.avg * 100:.3f}")
                
            # Validation loop
            with torch.no_grad():
                # Metrics
                val_loss.reset()
                val_accy.reset()
                
                model.train(mode=False)
                loop = tqdm(_val_load, position=0, leave=True)
                loop.set_description(f"epoch: {epoch_nb} / {20 - 1} - val")
                for imgs, lbls in loop:
                    # Set up the data and optimizer
                    imgs = transform(imgs.to(torch.float32).to(device))
                    lbls = lbls.to(device)
                    optimizer.zero_grad()
                    
                    # Forward pass and calculate loss
                    out = model(imgs)
                    lss = loss_func(out, lbls.max(dim=1)[1])
                    
                    # Update metrics 
                    val_loss.update(lss.item(), imgs.size()[0])
                    val_accy.update(torch.mean((out.max(dim=1)[1] == lbls.max(dim=1)[1]).to(torch.float32)), imgs.size()[0])
                     
                    loop.set_postfix_str(f"loss: {val_loss.avg:.3f} - accuracy: {val_accy.avg * 100:.3f}")
            del imgs, lbls
            torch.cuda.empty_cache()
    
        model.eval()
        print("\n\n")
        # Test loop
        with torch.no_grad():
            val_loss.reset()
            val_accy.reset()
            
            model.train(mode=False)
            loop = tqdm(_test_load, position=0, leave=True)
            loop.set_description("Testing: ")
            for imgs, lbls in loop:
                # Set up the data and optimizer
                imgs = transform(imgs.to(torch.float32).to(device))
                lbls = lbls.to(device)
                optimizer.zero_grad()
                
                # Forward pass and calculate loss
                out = model(imgs)
                lss = loss_func(out, lbls.max(dim=1)[1])
                
                # Update metrics 
                val_loss.update(lss.item(), imgs.size()[0])
                val_accy.update(torch.mean((out.max(dim=1)[1] == lbls.max(dim=1)[1]).to(torch.float32)), imgs.size()[0])
                
                loop.set_postfix_str(f"loss: {val_loss.avg:.3f} - accuracy: {val_accy.avg * 100:.3f}")
        del imgs, lbls
        torch.cuda.empty_cache()
    
    # Unfreeze the layers
    for param in model.parameters():
        param.requires_grad = True
    
    torch.save(model.state_dict(), PATH)
    
    model_FC = VGG19_FC()
    # print(model_FC)
    # Move model to device
    model_FC = model_FC.to(device)
    
    # Setting up optimizer and loss function
    optimizer = torch.optim.Adam(model_FC.parameters(), args.lr)
    loss_func = torch.nn.CrossEntropyLoss()
    
    PATH = "vgg_FC_mini_imagenet.pth"
    
    if(os.path.exists(PATH)):
        # Load model
        model_FC.set_weights(PATH)
        model_FC.to(device)
    else:
        # Training loop
        for epoch_nb in range(args.num_epochs):
            # Metrics
            trn_loss.reset()
            trn_accy.reset()
            
            # Training loop
            loop = tqdm(_train_load, position=0, leave=True)
            loop.set_description_str(f"Epoch: {epoch_nb}")
            for imgs, lbls in loop:
                imgs = transform(imgs.to(torch.float32).to(device))
                lbls = lbls.to(device)
                optimizer.zero_grad()
                
                # Forward pass
                out = model_FC(imgs).squeeze()
                lss = loss_func(out, lbls.max(dim=1)[1])
                
                # Backward pass
                lss.backward()
                optimizer.step()
                
                # Update metrics 
                trn_loss.update(lss.item())
                trn_accy.update(torch.mean((out.max(dim=1)[1] == lbls.max(dim=1)[1]).to(torch.float32)))
                
                loop.set_postfix_str(f"loss: {trn_loss.avg:.3f} - accuracy: {trn_accy.avg * 100:.3f}")
                
            # Validation loop
            with torch.no_grad():
                # Metrics
                val_loss.reset()
                val_accy.reset()
                
                model_FC.train(mode=False)
                loop = tqdm(_val_load, position=0, leave=True)
                loop.set_description(f"epoch: {epoch_nb} / {20 - 1} - val")
                for imgs, lbls in loop:
                    # Set up the data and optimizer
                    imgs = transform(imgs.to(torch.float32).to(device))
                    lbls = lbls.to(device)
                    optimizer.zero_grad()
                    
                    # Forward pass and calculate loss
                    out = model_FC(imgs).squeeze()
                    lss = loss_func(out, lbls.max(dim=1)[1])
                    
                    # Update metrics 
                    val_loss.update(lss.item(), imgs.size()[0])
                    val_accy.update(torch.mean((out.max(dim=1)[1] == lbls.max(dim=1)[1]).to(torch.float32)), imgs.size()[0])
                     
                    loop.set_postfix_str(f"loss: {val_loss.avg:.3f} - accuracy: {val_accy.avg * 100:.3f}")
            del imgs, lbls
            torch.cuda.empty_cache()
    
        model_FC.eval()
        print("\n\n")
        # Test loop
        with torch.no_grad():
            val_loss.reset()
            val_accy.reset()
            
            model_FC.train(mode=False)
            loop = tqdm(_test_load, position=0, leave=True)
            loop.set_description("Testing: ")
            for imgs, lbls in loop:
                # Set up the data and optimizer
                imgs = transform(imgs.to(torch.float32).to(device))
                lbls = lbls.to(device)
                optimizer.zero_grad()
                
                # Forward pass and calculate loss
                out = model(imgs).squeeze()
                lss = loss_func(out, lbls.max(dim=1)[1])
                
                # Update metrics 
                val_loss.update(lss.item(), imgs.size()[0])
                val_accy.update(torch.mean((out.max(dim=1)[1] == lbls.max(dim=1)[1]).to(torch.float32)), imgs.size()[0])
                
                loop.set_postfix_str(f"loss: {val_loss.avg:.3f} - accuracy: {val_accy.avg * 100:.3f}")
        del imgs, lbls
        torch.cuda.empty_cache()
    
    # Unfreeze the layers
    for param in model.parameters():
        param.requires_grad = True
    
    torch.save(model_FC.state_dict(), PATH)
    
    model_FC.eval()
    
    
    
    
    
    
    untransform = T.Compose([
        T.Normalize(mean=[0.0, 0.0, 0.0], std = [255., 255., 255.]),
        T.Normalize(mean=[-0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ])
    ])
    
    # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    img, lbl = _data[0]
    
    img = img[None, :].to(torch.float32)
    normed_img = transform(img)
    pred = model(normed_img)
    img = untransform(normed_img)
    
    gradcam = GradCAM.from_config(model_type='vgg', arch=model, layer_name='features_50')
    # get a GradCAM saliency map on the class index 10.
    #TODO: check logit
    mask, logit = gradcam(img, class_idx=torch.argmax(F.softmax(pred[0], dim=0), dim= 0))#np.argmax(lbl))
    # make heatmap from mask and synthesize saliency map using heatmap and img
    heatmap, cam_result = visualize_cam(mask, img)
    
    
    gradcampp = GradCAMpp.from_config(model_type='vgg', arch=model, layer_name='features_50')
    # get a GradCAM saliency map on the class index 10.
    maskPP, logitPP = gradcampp(img, class_idx=torch.argmax(F.softmax(pred[0], dim=0), dim= 0))#np.argmax(lbl))
    # make heatmap from mask and synthesize saliency map using heatmap and img
    heatmapPP, camPP_result = visualize_cam(maskPP, img)
    
    img = torch.swapaxes(img, 1, 3).squeeze()
    normed_img = torch.swapaxes(normed_img, 1, 3).squeeze()
    heatmap = torch.swapaxes(heatmap, 0, 2)
    cam_result = torch.swapaxes(cam_result, 0, 2)
    heatmapPP = torch.swapaxes(heatmapPP, 0, 2)
    camPP_result = torch.swapaxes(camPP_result, 0, 2)
    
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(img)
    plt.subplot(2, 3, 2)
    plt.imshow(heatmap)
    plt.subplot(2, 3, 3)
    plt.imshow(cam_result)
    plt.subplot(2, 3, 4)
    plt.imshow(img)
    plt.subplot(2, 3, 5)
    plt.imshow(heatmapPP)
    plt.subplot(2, 3, 6)
    plt.imshow(camPP_result)
    plt.show()