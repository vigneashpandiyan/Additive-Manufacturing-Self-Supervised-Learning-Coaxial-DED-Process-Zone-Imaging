# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 15:24:26 2022

@author: srpv
"""
import torch
import torch.nn as nn
import torchvision.transforms as torch_tf
import torchvision


class Solarization(nn.Module):
    def __call__(self, x):
        return torch_tf.functional.solarize(x, threshold=0.5)

def augmentation_tau_prime(p_blur=0.1, p_solarize=0.2):
    return nn.Sequential(
        torch_tf.RandomApply(torch.nn.ModuleList([torch_tf.RandomRotation(degrees=(0, 180),interpolation=torchvision.transforms.InterpolationMode.NEAREST,fill=0.065)]), p=0.75),
        torch_tf.RandomHorizontalFlip(p=0.75),
        torch_tf.RandomVerticalFlip(p=0.75),
        
        )
    

def augmentation_tau():
    return nn.Sequential(
        torch_tf.RandomApply(torch.nn.ModuleList([torch_tf.RandomRotation(degrees=(0, 90),interpolation=torchvision.transforms.InterpolationMode.NEAREST,fill=0.065)]), p=0.25),
        torch_tf.RandomHorizontalFlip(p=0.25),
        torch_tf.RandomVerticalFlip(p=0.25),
        
        )

