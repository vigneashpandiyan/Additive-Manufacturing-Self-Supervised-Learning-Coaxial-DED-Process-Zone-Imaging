# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 22:46:33 2023

@author: srpv
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch
from torch import nn
from einops import rearrange, repeat
import torchvision

from matplotlib import animation
from PIL import Image

from Utils import*
from Network import*
from Trainer import*
from Visualizing_Attention import*
#%%
# Data--> https://zenodo.org/records/10421423
# datadir = '../Data/train'   #place data inside 
#           '../Data/test'  
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

datadir ="C:/Users/srpv/Desktop/DED Byol/Data/"  #path towards the dataset
traindir = datadir + 'train/'
testdir = datadir + 'test/'

batch_size = 256
n_epochs=100
lr=1e-3
#%% Dataset preparation / Data loader

data_transform = transforms.Compose([
        torchvision.transforms.Resize((480,320)),
        transforms.RandomHorizontalFlip(p=0.5), #Augmentation to the dataset
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor()])


train_dataset = datasets.ImageFolder(root=traindir,transform=data_transform)
test_dataset = datasets.ImageFolder(root=testdir,transform=data_transform)

n_classes = len(train_dataset.classes)
classes = train_dataset.classes

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


model = ViT(patch_size=20, dim=32, depth=3, heads=8, mlp_dim=32)

if device:
    model.to(device)


trained_model, train_loss_history, valid_loss_history, valid_accuracy_history,Learning_rate = run_training_loop(model,train_loader,valid_loader,n_epochs,lr,device)
plots(train_loss_history,valid_loss_history,Learning_rate)

#%%

test_performance(model,valid_loader,device)


# In[ ]:


trained_model.eval()
with torch.no_grad():
  for data, target in valid_loader:
      data,target = data.to(device,dtype=torch.float),target.to(device,dtype=torch.long)
      output, attn_matrix,_ = trained_model(data, return_attn_matrix = True)
      break

matrix = []
for i in attn_matrix:
  i = torch.mean(i,0)
  i=i.detach().cpu().numpy() 
  matrix.append(i)
  
fig, ax = plt.subplots(2, 4, figsize=(12,6))
for i in np.arange(2):
    for j in np.arange(4):
        ax[i,j].imshow(matrix[j+4*i],cmap = "Purples")
fig.suptitle("validation attention weights for 8 heads")
plt.savefig("Training_heads.png",bbox_inches='tight',dpi=200)
plt.show()

    
#%%
count_parameters(trained_model)
classes = ('P1', 'P2', 'P3', 'P4')

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)


_, _,X_train, y_train=Compute_latents(trained_model,train_loader,device,'train')
y_true, y_pred,X_test, y_test=Compute_latents(trained_model,valid_loader,device,'test')

plotname= 'CNN_Univariate'+'_confusion_matrix'+'.png'
plot_confusion_matrix(y_true, y_pred,classes,plotname)


#%%

graph_name='CNN_TSNE_Latent' +'_'+'.png'
ax,fig=Three_embeddings(X_test, y_test,graph_name,ang=195)#135
gif_name= str('CNN_TSNE_Latent')+'.gif'

def rotate(angle):
      ax.view_init(azim=angle)
      
angle = 3
ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
ani.save(gif_name, writer=animation.PillowWriter(fps=20))


graph_name_2D='CNN_Training_Feature_2D' +'_'+'.png'
graph_title = "Feature space distribution"
plot_embeddings(X_test, y_test,classes,graph_title,graph_name_2D, xlim=None, ylim=None)


graph_name_3D='CNN_Latent_3D' +'_'+'.png'
ax,fig=Three_Latent_embeddings(X_test, y_test,graph_name_3D,ang=195, dim_1=20, dim_2=30, dim_3=31)#135
gif2_name= str('CNN_3D_Latent')+'.gif'
angle = 3
ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
ani.save(gif2_name, writer=animation.PillowWriter(fps=20))


Distribution_name_2D='CNN_Distribution_2D_'
distribution_plots(X_test, y_test,Distribution_name_2D)


#%%

fig, axs = plt.subplots(
  nrows=8,
  ncols=4,
  sharey=False,
  figsize=(11, 14),
  dpi=1000
)

columns = np.atleast_2d(X_train).shape[1]
graph_name_32D='CNN_Latent_32D' +'_'+'.png'
for i in range(columns):
  ax = axs.flat[i]
  Cummulative_plots(X_train, y_train,i,ax)
  
fig.tight_layout();
fig.savefig(graph_name_32D)
fig.show()

#%%

from Classifier import*

# X_train, X_test, y_train, y_test = train_test_split( embeddings,labels, test_size=0.33, random_state=42)

classifier(X_train, X_test, y_train, y_test,'LogisticRegression','CNN')
classifier(X_train, X_test, y_train, y_test,'SVM','CNN')
classifier(X_train, X_test, y_train, y_test,'RF','CNN')
classifier(X_train, X_test, y_train, y_test,'GaussianNB','CNN')

#%%


from torch.utils.data import Subset, DataLoader
subsets = {target: Subset(test_dataset, [i for i, (x, y) in enumerate(test_dataset) if y == target]) for _, target in test_dataset.class_to_idx.items()}
loaders = {target: DataLoader(subset) for target, subset in subsets.items()}
trained_model.eval()
total_classes=len(test_dataset.class_to_idx.items())
patch_size=20
for x in range(total_classes):
    print(x)
    class_loader = loaders[x]
    with torch.no_grad():
        for data, target in class_loader:
            data,target = data.to(device,dtype=torch.float),target.to(device,dtype=torch.long)
            # print("data shape...",data.shape)
            output, attn_matrix,_ = trained_model(data, return_attn_matrix = True)
            # print(attn_matrix.shape)
            data=data.squeeze()
            visualize_attention(trained_model, data, patch_size,x, device)
            
            break
    print(attn_matrix.shape)
    matrix = []
    for i in attn_matrix:
      i = torch.mean(i,0)
      i=i.detach().cpu().numpy() 
      matrix.append(i)
    # print(matrix.shape)  
    fig, ax = plt.subplots(2, 4, figsize=(12,6))
    for i in np.arange(2):
        for j in np.arange(4):
            ax[i,j].imshow(matrix[j+4*i],cmap = "bwr")
            
    fig.suptitle("validation attention weights for 8 heads on category "+str(x))
    graphname="Training_heads "+str(x)+".png"
    plt.savefig(graphname,bbox_inches='tight',dpi=200)
    plt.show()
