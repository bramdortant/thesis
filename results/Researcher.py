# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 03:30:55 2024

@author: Bram
"""

import numpy as np
import torch
from tqdm import tqdm

import torch.nn.functional as F
from GradCAM import GradCAM, GradCAMpp, visualize_cam
from captum.attr import IntegratedGradients, GuidedBackprop

class Researcher(object):
    """ This is a researcher object to produce results explained in: """ 
    #TODO: add

    def __init__(self, device, model, model_FC, train_load, val_load, test_load, transform) -> None:
        """ """
        super(Researcher, self).__init__()
        
        self.device = device
        self.model = model
        self.model_FC = model_FC
        self.train_load = train_load
        self.val_load = val_load
        self.test_load =  test_load
        self.transform = transform
        
    def research(self) -> torch.tensor:
        model = self.model.to(self.device).eval()
        # model.eval()
        model_FC = self.model_FC.to(self.device).eval()
        # model_FC.eval()
        
        # Unfreeze the layers
        for param in model.parameters():
            param.requires_grad = True
            
        # Unfreeze the layers
        for param in model_FC.parameters():
            param.requires_grad = True
        
        with torch.no_grad():
           
            # training loop
            loop = tqdm(self.train_load, position=0, leave=True)
            loop.set_description_str("Reviewing Training data")
            win_counter = torch.zeros(4)
            drop_counter = torch.zeros(4)
            increase_counter = torch.zeros(4)
            for imgs, lbls in loop:
                for img, lbl in zip(imgs, lbls):
                    img = self.transform(img.to(torch.float32).unsqueeze(0).to(self.device))
                    img.requires_grad = True
                    lbl = lbl.to(self.device)
                    
                    out = model(img)
                    # print(out)
                    print(out.requires_grad)
                    
                    win = self.win_percentage(img, model, model_FC, lbl)
                    drop = self.drop_percentage(img, model, model_FC, lbl)
                    increase = self.increase_percentage(img, model, model_FC, lbl)
                    
                    win_counter += win
                    drop_counter += drop
                    increase_counter += increase
                
            training = (win_counter, drop_counter, increase_counter)
            
            # validation loop
            loop = tqdm(self.val_load, position=0, leave=True)
            loop.set_description_str("Reviewing Validation data")
            win_counter = torch.zeros(4)
            drop_counter = torch.zeros(4)
            increase_counter = torch.zeros(4)
            for imgs, lbls in loop:
                for img, lbl in imgs, lbls:
                    img = self.transform(img.to(torch.float32).to(self.device))
                    img.requires_grad = True
                    lbl = lbl.to(self.device)
                    
                    win = self.win_percentage(img, model, model_FC, lbl)
                    drop = self.drop_percentage(img, model, model_FC, lbl)
                    increase = self.increase_percentage(img, model, model_FC, lbl)
                    
                    win_counter += win
                    drop_counter += drop
                    increase_counter += increase
                
            validation = (win_counter, drop_counter, increase_counter)
            
            # Test loop
            loop = tqdm(self.test_load, position=0, leave=True)
            loop.set_description_str("Reviewing Test data")
            win_counter = torch.zeros(4)
            drop_counter = torch.zeros(4)
            increase_counter = torch.zeros(4)
            for imgs, lbls in loop:
                for img, lbl in imgs, lbls:
                    img = self.transform(img.to(torch.float32).to(self.device))
                    img.requires_grad = True
                    lbl = lbl.to(self.device)
                    
                    win = self.win_percentage(img, model, model_FC, lbl)
                    drop = self.drop_percentage(img, model, model_FC, lbl)
                    increase = self.increase_percentage(img, model, model_FC, lbl)
                    
                    win_counter += win
                    drop_counter += drop
                    increase_counter += increase
            
            test = (win_counter, drop_counter, increase_counter)
            
        return training, validation, test
    
    
    def CAM(self, model_FC, normed_img, lbl):
        # initialize a model, model_dict and gradcam
        gradcam = GradCAM.from_config(model_type='vgg', arch=model_FC, layer_name='FC_features_50')
        # get a GradCAM saliency map on the class index lbl.
        mask, logit = gradcam(normed_img, class_idx=torch.argmax(lbl))
        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, normed_img)
        heatmap = np.transpose(heatmap, (1,2,0))
        
        return heatmap, mask.squeeze()
    
    
    def Grad_CAMPP(self, model, normed_img, lbl):
        gradcampp = GradCAMpp.from_config(model_type='vgg', arch=model, layer_name='features_50')
        # get a GradCAM saliency map
        maskPP, logitPP = gradcampp(normed_img, class_idx=torch.argmax(lbl))
        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmapPP, camPP_result = visualize_cam(maskPP, normed_img)
        heatmapPP = np.transpose(heatmapPP, (1,2,0))
        
        return heatmapPP, maskPP.squeeze()
    
    
    def integrated_gradients(self, model, normed_img, lbl, outlier_perc):
        ig = IntegratedGradients(model)
        model.zero_grad()
        attributions_ig = ig.attribute(normed_img, target=torch.argmax(lbl), baselines=normed_img*0)
        
        if torch.max(np.transpose(normed_img.squeeze().detach(), (1,2,0))) <= 1.0:
            normed_img = np.clip((np.transpose(normed_img, (1,2,0))* 255).astype(int), 0, 255)
        
        attr_combined = np.sum(np.transpose(attributions_ig.squeeze().detach().numpy(), (1,2,0)), axis=2)
        attr_combined = (attr_combined > 0) * attr_combined
        
        sorted_vals = np.sort(attr_combined.flatten())
        cum_sums = np.cumsum(sorted_vals)
        threshold_id = np.where(cum_sums >= cum_sums[-1] * 0.01 * (100 - outlier_perc))[0][0]
        threshold = sorted_vals[threshold_id]
        
        attr_norm = attr_combined / threshold
        mask_ig = np.clip(attr_norm, -1, 1)
        
        heatmap_ig, ig_result = visualize_cam(mask_ig, normed_img)
        heatmap_ig = np.transpose(heatmap_ig, (1,2,0))
       
        return heatmap_ig, torch.from_numpy(mask_ig)
        
    
    def guided_gradcam(self, model, normed_img, lbl, outlier_perc):
        gbp = GuidedBackprop(model)
        model.zero_grad()
        
        attributions_gbp = gbp.attribute(normed_img, target=torch.argmax(lbl))
        
        if torch.max(np.transpose(normed_img.squeeze().detach(), (1,2,0)) <= 1.0):
            normed_img = np.clip((np.transpose(normed_img, (1,2,0))* 255).astype(int), 0, 255)
        
        
        attr_combined = np.sum(np.transpose(attributions_gbp.squeeze().detach().numpy(), (1,2,0)), axis=2)
        attr_combined = (attr_combined > 0) * attr_combined
        
        sorted_vals = np.sort(attr_combined.flatten())
        cum_sums = np.cumsum(sorted_vals)
        threshold_id = np.where(cum_sums >= cum_sums[-1] * 0.01 * (100 - outlier_perc))[0][0]
        threshold = sorted_vals[threshold_id]
        
        attr_norm = attr_combined / threshold
        mask_gb = np.clip(attr_norm, -1, 1)
        heatmap_gbp, gbp_result = visualize_cam(mask_gb, normed_img)
        heatmap_gbp = np.transpose(heatmap_gbp, (1,2,0))
        
        # initialize a model, model_dict and gradcam
        gcam = GradCAM.from_config(model_type='vgg', arch=model, layer_name='features_50')
        # get a GradCAM saliency map on the class index 10.
        mask_gradcam, logit = gcam(normed_img, class_idx=torch.argmax(lbl))
        # make heatmap from mask and synthesize sal.iency map using heatmap and img
        heatmapGC, Gcam_result = visualize_cam(mask_gradcam, normed_img)
        heatmapGC = np.transpose(heatmapGC, (1,2,0))
        
        mask_gg = torch.from_numpy(mask_gb) * mask_gradcam
        mask_gg /= torch.max(mask_gg)
        
        guided_gradcam, gg_result = visualize_cam(mask_gg, normed_img)
        
        return heatmap_gbp, torch.from_numpy(mask_gb), np.transpose(guided_gradcam.squeeze().detach(),(1,2,0)), mask_gg.squeeze()
    
    def mask(img, heatmap, cut_off):
        # https://stackoverflow.com/questions/74771948/how-to-localize-red-regions-in-heatmap-using-opencv
        
        max_val = 255 if torch.max(heatmap) > 1 else 1
        prediction = heatmap/max_val
        mask = prediction[:,:,None] > cut_off
        
        return np.multiply(mask, img)
    
    def output_and_confidence(self, model, normed_img):
        output = model(normed_img)
        probs = F.softmax(output, dim=1)
        conf, classes = torch.max(probs,1)
        
        return classes, conf
    
    def drop_percentage(self, normed_img, model, model_FC, lbl):
        pred, conf = self.output_and_confidence(model, normed_img)
        pred_FC, conf_FC = self.output_and_confidence(model_FC, normed_img)
        
        heatmap, mask_cam = self.CAM(model_FC, normed_img, lbl)
        heatmapPP, mask_PP = self.Grad_CAMPP(model, normed_img, lbl)
        IG, mask_ig = self.integrated_gradients(model, normed_img, lbl)
        guided_bp, guided_gcam = self.guided_gradcam(model, normed_img, lbl)
        
        CAM_mask = torch.max(0, conf_FC - self.output_and_confidence(model_FC, np.transpose(self.mask(normed_img, mask_cam), (2,0,1)).unsqueeze())[1])
        grad_camPP_mask = torch.max(0, conf - self.output_and_confidence(model, np.transpose(self.mask(normed_img, mask_PP), (2,0,1)).unsqueeze())[1])
        integrated_gradients_mask = torch.max(0, conf - self.output_and_confidence(model, np.transpose(self.mask(normed_img, mask_ig), (2,0,1)).unsqueeze())[1])
        guided_gradcam_mask = torch.max(0, conf - self.output_and_confidence(model, model(self.mask(normed_img, guided_gcam)))[1])
        
        return CAM_mask, grad_camPP_mask, integrated_gradients_mask, guided_gradcam_mask
        
    def increase_percentage(self, normed_img, model, model_FC, lbl):
        pred, conf = self.output_and_confidence(model, normed_img)
        pred_FC, conf_FC = self.output_and_confidence(model_FC, normed_img)
        
        heatmap, mask_cam = self.CAM(model_FC, normed_img, lbl)
        heatmapPP, mask_PP = self.Grad_CAMPP(model, normed_img, lbl)
        IG, mask_ig = self.integrated_gradients(model, normed_img, lbl)
        guided_bp, guided_gcam = self.guided_gradcam(model, normed_img, lbl)
        
        CAM_mask = conf_FC < self.output_and_confidence(model_FC, np.transpose(self.mask(normed_img, mask_cam), (2,0,1)).unsqueeze())[1]
        grad_camPP_mask = conf < self.output_and_confidence(model, np.transpose(self.mask(normed_img, mask_PP), (2,0,1)).unsqueeze())[1]
        integrated_gradients_mask = conf < self.output_and_confidence(model, np.transpose(self.mask(normed_img, mask_ig), (2,0,1)).unsqueeze())[1]
        guided_gradcam_mask = conf < self.output_and_confidence(model, model(self.mask(normed_img, guided_gcam)))[1]
        
        return CAM_mask, grad_camPP_mask, integrated_gradients_mask, guided_gradcam_mask


    def win_percentage(self, normed_img, model, model_FC, lbl):
        heatmap, mask_cam = self.CAM(model_FC, normed_img, lbl)
        heatmapPP, mask_PP = self.Grad_CAMPP(model, normed_img, lbl)
        IG, mask_ig = self.integrated_gradients(model, normed_img, lbl)
        guided_bp, guided_gcam = self.guided_gradcam(model, normed_img, lbl)
        
        CAM_mask = self.output_and_confidence(model_FC, np.transpose(self.mask(normed_img, mask_cam), (2,0,1)).unsqueeze())[1]
        grad_camPP_mask = self.output_and_confidence(model, np.transpose(self.mask(normed_img, mask_PP), (2,0,1)).unsqueeze())[1]
        integrated_gradients_mask = self.output_and_confidence(model, np.transpose(self.mask(normed_img, mask_ig), (2,0,1)).unsqueeze())[1]
        guided_gradcam_mask = self.output_and_confidence(model, model(self.mask(normed_img, guided_gcam)))[1]
        
        array = [CAM_mask, grad_camPP_mask, integrated_gradients_mask, guided_gradcam_mask]
        # https://stackoverflow.com/questions/5284646/rank-items-in-an-array-using-python-numpy-without-sorting-array-twice
        order = np.argsort(np.argsort(array))
        
        return order
    
  
