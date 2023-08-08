# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 15:56:27 2022

@author: srpv

"""
## Importing required libraries

import os
import torch
import torchvision
import time
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import STL10
from torchvision.transforms import ToTensor
import torch
import torch.nn as nn
import torchvision.transforms as torch_tf
import torch.nn.functional as f
from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
#%%

class BYOL(nn.Module):
    def __init__(self,
                 model,
                 augmentation,
                 augmentation_prime,
                 embedding,
                 epoch,num_step,
                 projection_size,
                 tau = 0.999):
        super().__init__()    
        
        self.encoder_online = model
        self.projector_online = BYOL.mlp(embedding, projection_size) # Create a network for projection
        self.online_common = nn.Sequential(self.encoder_online,
                                            self.projector_online)  # Encoder and projector are the same for target and online network
        
        
        self.predictor_online = BYOL.mlp(projection_size, projection_size)
        self.online = nn.Sequential(self.online_common, self.predictor_online) # Whole online network
        self.target = deepcopy(self.online_common) # Target network without prediction head

        self.tau = tau # Tau for moving exponential average
        self.augmentation = augmentation
        self.augmentation_prime = augmentation_prime

        self.loss_fn = nn.MSELoss(reduction="sum") # Loss function for comparision of outputs of two networks
        self.optimiser = optim.AdamW(self.online.parameters(),lr=0.01) # Optimizer, diffrent than in paper
        # self.optimiser = torch.optim.SGD(self.online.parameters(),lr=0.01,momentum=0.9)
        self.scheduler = StepLR(self.optimiser, step_size = 20, gamma= 0.50 )
        self.epochs=epoch
        # self.scheduler = CosineAnnealingLR(self.optimiser, epoch, eta_min=1e-3)

    @staticmethod
    
    def mlp(dim_in: int, projection_size: int) -> nn.Module:
        
        return nn.Sequential(nn.Linear(dim_in, projection_size)) # and a final linear layer with output dimension 256

    def fit(self, train_dl: DataLoader, val_dl: DataLoader) -> dict:
        
        results = {"train_loss": [], "val_loss": []}

        for epoch in range(self.epochs):
            start = time.time()
            
            train_loss = self.train_one_epoch(train_dl) # Train (fit) model on unlabelled data
            
            self.scheduler.step()
            lr_rate = self.scheduler.get_last_lr()[0]
            print("learning rate",lr_rate)
            
            val_loss = self.validate(val_dl) # Validate on validation data (labels omitted)

            # Print results
            print(f"Epoch {epoch+1}: train loss = {train_loss:.4f}, validation "
                f"loss = {val_loss:.4f}, time "
                f"{((time.time() - start)/60):.1f} min")
            
            results["train_loss"].append(float(train_loss))
            results["val_loss"].append(float(val_loss))

        return results

    def train_one_epoch(self,  train_dl: DataLoader):
        
        self.online.train()
        self.target.train()

        for X_batch, y_batch in tqdm(train_dl):
            

            X_batch, y_batch = X_batch.cuda(), y_batch.cuda()
            loss = self.forward(X_batch)
            loss.backward()
            self.optimiser.step()
            self.optimiser.zero_grad() 
            self.update_target()
            
            
        self.target.eval()
        self.online.eval()
        loss = 0
        all = 0
        
        for X_batch, y_batch in tqdm(train_dl):
            X_batch, y_batch = X_batch.cuda(), y_batch.cuda()
            with torch.no_grad():
                loss += self.forward(X_batch)
            all += len(X_batch)
            
        return loss / all

        
    def forward(self,
                X_batch: torch.Tensor
        ) -> torch.Tensor:
        with torch.no_grad():
            v, v_prime = self.augmentation(X_batch), self.augmentation_prime(X_batch)
            
        pred = self.online(v)
        pred = f.normalize(pred)
        with torch.no_grad():
            z = self.target(v_prime)
            z = f.normalize(z)
    
        return self.loss_fn(pred, z)
    
    
    def update_target(self):
        
        for p_online, p_target in zip(self.online_common.parameters(),
                                      self.target.parameters()):
            p_target.data = self.tau * p_target.data + (1 - self.tau) * p_online.data


    def validate(self, dataloader: DataLoader) -> torch.Tensor:
        
        loss = 0
        all = 0
        for X_batch, y_batch in tqdm(dataloader):
            X_batch, y_batch = X_batch.cuda(), y_batch.cuda()
            with torch.no_grad():
                loss += self.forward(X_batch)
            all += len(X_batch)
            
        return loss / all


    def get_embedding(self, X_batch: torch.Tensor) -> torch.Tensor:
        
        return self.online(X_batch)

