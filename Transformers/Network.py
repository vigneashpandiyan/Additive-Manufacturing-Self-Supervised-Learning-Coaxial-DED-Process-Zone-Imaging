# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 11:47:42 2022
@author: srpv

"""

import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import torch
import numpy as np


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        print(x.shape)
        return x




class SingleHeadAttention(nn.Module):
    def __init__(self, input_dim, inner_dim, dropout = 0.1):
        super().__init__()
        # TODO
        self.q = nn.Linear(input_dim, inner_dim)
        self.k = nn.Linear(input_dim, inner_dim)
        self.v = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.d_k = inner_dim

    def forward(self, x):
        # TODO
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        attn_weights = self.softmax(torch.matmul(Q, torch.transpose(K, -2, -1)) / np.sqrt(self.d_k))
        out = torch.matmul(attn_weights, V)
        
        out = self.dropout(out)

        return out, attn_weights


# In[ ]:


# model.transformer[5].attn[1].attention_heads[0].attentionmatrix


# Test the following multihead attention implementation that relies on the single-head attention implementation above.

# In[ ]:


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        self.attention_heads = nn.ModuleList([
            SingleHeadAttention(dim, dim_head, dropout=dropout)
            for _ in range(heads)
        ])

    def forward(self, x):
        outputs = [head(x) for head in self.attention_heads]
        out = torch.cat([output[0] for output in outputs], dim=-1)
        attn_matrix = torch.stack([output[1] for output in outputs])
        
        # print(attn_matrix.shape)
        # print(out.shape)
        return out, attn_matrix


# Test code for multihead attention:


# In[ ]:


class ViTLayer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.attn = nn.Sequential(
            # TODO
            nn.LayerNorm(dim),
            Attention(dim=dim,heads=heads, dim_head=dim_head, dropout=dropout),
        )
        self.feedforward = nn.Sequential(
            # TODO
            nn.LayerNorm(dim),
            #A
            nn.Linear(dim,dim),
            nn.Dropout(p=dropout),
            #Relu
            nn.ReLU(),
            #B
            # nn.Linear(mlp_dim,dim),
            #dropout
            # nn.Dropout(p=dropout)
        )
        self.attn_project = nn.Linear(heads*dim_head,dim)

    def forward(self, x):
        # TODO
        # print("x....1",x.shape)
        outattn, attn_matrix = self.attn(x)
        # print("outattn....1",outattn.shape)
        # print("attn_matrix....1",attn_matrix.shape)
        outattn = self.attn_project(outattn) + x
        #apply feedforward
        
        # print("outattn....2",outattn.shape)
        outmlp = self.feedforward(outattn)
        out = outattn + outmlp
        return out, attn_matrix
  


# ## ViT code for 3 (c)

# In[ ]:


class ViT(nn.Module):
    def __init__(self, patch_size=20, dim=32, depth=6, heads=8, mlp_dim=128, 
                 dim_head = 64,
                 dropout = 0., emb_dropout = 0.):
        super().__init__()

        image_height, image_width = 480, 320
        num_classes = 4
        channels = 3
        
        patch_height = patch_size
        patch_width = patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0, (
            'Image dimensions must be divisible by the patch size.')

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width


        self.dim=dim
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.transformer = nn.Sequential(*nn.ModuleList([
            ViTLayer(dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)                       
        ]))

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
        self.adaptivepooling=nn.AdaptiveAvgPool1d(1)

    def forward(self, img, return_attn_matrix = False):
        
        
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :(n)]
        x = self.emb_dropout(x)
    
        for layer in self.transformer:
           
          x, attn_matrix = layer(x)
          
        x=x.transpose(2,1)
        x = self.adaptivepooling(x)
        # x = x[:, 0]
        x=x.view(x.size(0), -1)
        
        if return_attn_matrix:
          return self.mlp_head(x), attn_matrix,x
        else:
          return self.mlp_head(x)



# T = 5
# input_dim = 32


# heads = 8
# dim_head = 64

# test_input = torch.zeros((batch_size, T, input_dim))
# print(test_input.shape)


# test_attention_module = Attention(input_dim)
# test_output, test_attn_matrix = test_attention_module(test_input)

# # print(test_output)
# assert test_output.size() == (batch_size, T, heads * dim_head), "Shapes are incorrect"
