#!/usr/bin/env python
# coding: utf-8


import os
import torch
import numpy as np
import math
from functools import partial
import torch
import torch.nn as nn

import ipywidgets as widgets
import io
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch import nn

import warnings
warnings.filterwarnings("ignore")


# ## Vision Transformer

# In[3]:

    
def visualize_attention(trained_model, img, patch_size,name, device):
    # make the image divisible by the patch size
    
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - \
        img.shape[2] % patch_size
    img_1 = img[:, :w, :h].unsqueeze(0)

    w_featmap = img_1.shape[-2] // patch_size
    h_featmap = img_1.shape[-1] // patch_size
    output, attentions,latent = trained_model(img_1.to(device), return_attn_matrix = True)
    attentions=attentions.transpose(1,0)
    nh = attentions.shape[1]  # number of head
    attentions = attentions[0, :, 0, :]
    attentions = attentions.reshape(nh, h_featmap, w_featmap)
    attentions=attentions.cpu().detach()
    
    attentions = nn.functional.interpolate(attentions.unsqueeze(
        0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()
    
    plot_attention(img, attentions,name)
    
    
def plot_attention(img, attention,name):

    n_heads = attention.shape[0]
    plt.figure(figsize=(12,6))
    for i in range(n_heads):
        plt.subplot(n_heads//4, 4, i+1)
        plt.imshow(attention[i], cmap='inferno')
        plt.title(f"Head n: {i+1}")
    plt.tight_layout()
    plt.savefig("Attention of category "+str(name)+".png",bbox_inches='tight',dpi=200)
    plt.show()
    
    img=img.permute(2,1,0)
    img=img.detach().cpu().numpy()
    # img=img[0][0][0].numpy()
    # attention=attention.detach().cpu().numpy()
    plt.figure(figsize=(6, 10))
    text = ["Original Image", "Head Mean"]
    for i, fig in enumerate([img, np.mean(attention, 0)]):
        plt.subplot(1, 2, i+1)
        plt.imshow(fig, cmap='inferno')
        plt.title(text[i])
    plt.tight_layout()
    plt.savefig("Cumulative Attention of category "+str(name)+".png",bbox_inches='tight',dpi=200)
    plt.show()


