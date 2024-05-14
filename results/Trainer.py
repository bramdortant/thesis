# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 02:13:29 2024

@author: Bram
"""

import torch
from tqdm import tqdm

from EarlyStopping import EarlyStopping
from AverageMeter import EpochAverageMeter, RunningAverageMeter



class Trainer(object):
    """ This is a trainer object used to train a VGG19 network"""

    def __init__(self, device, model, train_load, val_load, transform, augment_transform, lr, PATH) -> None:
        """ """
        super(Trainer, self).__init__()
        
        self.device = device
        self.model = model
        self.train_load = train_load
        self.val_load = val_load
        self.transform = transform
        self.augment_transform = augment_transform
        self.lr = lr
        self.PATH = PATH
        
        
    def train(self, num_epochs, finetune_epochs) -> None :
        model = self.model.to(self.device)
        trn_loss = RunningAverageMeter()
        trn_accy = RunningAverageMeter()
        val_loss = EpochAverageMeter()
        val_accy = EpochAverageMeter()
        
        # Setting up optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), self.lr)
        loss_func = torch.nn.CrossEntropyLoss()
        
        
        # Early stopping for training loop
        early_stopping = EarlyStopping(verbose=True)
        
        # Training loop
        for epoch_nb in range(num_epochs):
            # Metrics
            trn_loss.reset()
            trn_accy.reset()
            
            # Training loop
            loop = tqdm(self.train_load, position=0, leave=True)
            loop.set_description_str(f"Epoch: {epoch_nb}")
            for imgs, lbls in loop:
                imgs = self.augment_transform(imgs.to(torch.float32).to(self.device))
                lbls = lbls.to(self.device)
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
                loop = tqdm(self.val_load, position=0, leave=True)
                loop.set_description(f"epoch: {epoch_nb} / {num_epochs - 1} - val")
                for imgs, lbls in loop:
                    # Set up the data and optimizer
                    imgs = self.transform(imgs.to(torch.float32).to(self.device))
                    lbls = lbls.to(self.device)
                    optimizer.zero_grad()
                    
                    # Forward pass and calculate loss
                    out = model(imgs)
                    lss = loss_func(out, lbls.max(dim=1)[1])
                    
                    # Update metrics 
                    val_loss.update(lss.item(), imgs.size()[0])
                    val_accy.update(torch.mean((out.max(dim=1)[1] == lbls.max(dim=1)[1]).to(torch.float32)), imgs.size()[0])
                     
                    loop.set_postfix_str(f"loss: {val_loss.avg:.3f} - accuracy: {val_accy.avg * 100:.3f}")
            early_stopping(val_loss.avg, model)
            if early_stopping.early_stop:
                print("Early Stop!")
                break
                
            
            del imgs, lbls
            torch.cuda.empty_cache()
            
        model.load_state_dict(torch.load('checkpoint.pt'))
            
        # Setting up optimizer and loss function for finetuning
        optimizer = torch.optim.Adam(model.parameters(), self.lr/10.0)
        loss_func = torch.nn.CrossEntropyLoss()
        
        # Unfreeze the layers for finetuning
        for param in model.parameters():
            param.requires_grad = True
            
        # Early stopping for finetuning loop
        min_val_loss = early_stopping.get_val_loss_min()
        early_stopping = EarlyStopping(verbose=True)
        early_stopping(min_val_loss, model)
            
        # Fine tune loop
        for epoch_nb in range(finetune_epochs):
            # Metrics
            trn_loss.reset()
            trn_accy.reset()
            
            # Training loop
            loop = tqdm(self.train_load, position=0, leave=True)
            loop.set_description_str(f"Epoch: {epoch_nb}")
            for imgs, lbls in loop:
                imgs = self.augment_transform(imgs.to(torch.float32).to(self.device))
                lbls = lbls.to(self.device)
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
                loop = tqdm(self.val_load, position=0, leave=True)
                loop.set_description(f"epoch: {epoch_nb} / {finetune_epochs - 1} - val")
                for imgs, lbls in loop:
                    # Set up the data and optimizer
                    imgs = self.transform(imgs.to(torch.float32).to(self.device))
                    lbls = lbls.to(self.device)
                    optimizer.zero_grad()
                    
                    # Forward pass and calculate loss
                    out = model(imgs)
                    lss = loss_func(out, lbls.max(dim=1)[1])
                    
                    # Update metrics 
                    val_loss.update(lss.item(), imgs.size()[0])
                    val_accy.update(torch.mean((out.max(dim=1)[1] == lbls.max(dim=1)[1]).to(torch.float32)), imgs.size()[0])
                     
                    loop.set_postfix_str(f"loss: {val_loss.avg:.3f} - accuracy: {val_accy.avg * 100:.3f}")
            early_stopping(val_loss.avg, model)
            if early_stopping.early_stop:
                print("Early Stop!")
                break
                
            
            del imgs, lbls
            torch.cuda.empty_cache()
        
        self.model.load_state_dict(torch.load('checkpoint.pt'))
        torch.save(self.model.state_dict(), self.PATH)
        
        # Unfreeze the layers
        for param in model.parameters():
            param.requires_grad = True
    
    def test(self, test_load) -> None :
        model = self.model.to(self.device)
        model.eval()
        
        
        val_loss = EpochAverageMeter()
        val_accy = EpochAverageMeter()
        
        # Setting up optimizer and loss function for finetuning
        optimizer = torch.optim.Adam(model.parameters(), self.lr)
        loss_func = torch.nn.CrossEntropyLoss()
        
        # Test loop
        predictions = []
        with torch.no_grad():
            val_loss.reset()
            val_accy.reset()
           
            model.train(mode=False)
            loop = tqdm(test_load, position=0, leave=True)
            loop.set_description("Testing: ")
            for imgs, lbls in loop:
                # Set up the data and optimizer
                imgs = self.transform(imgs.to(torch.float32).to(self.device))
                lbls = lbls.to(self.device)
                optimizer.zero_grad()
               
                # Forward pass and calculate loss
                out = model(imgs)
                for pred in out:
                    predictions.append(torch.argmax(pred))
                lss = loss_func(out, lbls.max(dim=1)[1])
               
                # Update metrics 
                val_loss.update(lss.item(), imgs.size()[0])
                val_accy.update(torch.mean((out.max(dim=1)[1] == lbls.max(dim=1)[1]).to(torch.float32)), imgs.size()[0])
               
                loop.set_postfix_str(f"loss: {val_loss.avg:.3f} - accuracy: {val_accy.avg * 100:.3f}")
        del imgs, lbls
        torch.cuda.empty_cache()
        
        # print(type(predictions))
        # return predictions
       
        # cm = confusion_matrix(_test.tst_lbls.tolist(), predictions)
        # ConfusionMatrixDisplay(cm).plot()
