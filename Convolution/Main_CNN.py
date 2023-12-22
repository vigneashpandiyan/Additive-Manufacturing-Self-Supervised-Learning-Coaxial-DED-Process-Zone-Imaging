# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 11:47:42 2022
@author: srpv

The purpose of this script is to train the
CNN 

"""

#%% Libraries to import
import numpy as np
import matplotlib.pyplot as plt

import torchvision
from torchvision import transforms
from torchvision import datasets
import torch
import torch as nn
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from matplotlib import animation
from Metrics import AccumulatedAccuracyMetric
from Trainer import fit
from Network import *
from Utils import*
from Classifier import*
cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Setting Up CUDA

#%%
# Data--> https://zenodo.org/records/10421423
# datadir = '../Data/train'   #place data inside 
#           '../Data/test'  

datadir ="C:/Users/srpv/Desktop/DED Byol/Data/"  #path towards the dataset
traindir = datadir + 'train/'
testdir = datadir + 'test/'

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

# Set up data loaders
batch_size = 256
kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

#%% Model initialization

embeddinglayer_size=64
projection_size=32
model = EmbeddingNet(dropout=0.05,embedding=embeddinglayer_size,projection_size=projection_size)
model = ConvNet(model,projection_size=projection_size,
                n_classes=len(train_dataset.classes))

count_parameters(model)

if cuda:
    model.cuda()

loss_fn = nn.CrossEntropyLoss(reduction='mean')

n_epochs = 100
log_interval = 25
# optimizer =  torch.optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
optimizer =  torch.optim.AdamW(model.parameters(),lr=0.01)
scheduler = StepLR(optimizer, step_size = 20, gamma= 0.50 )

#%% Model training

train_losses,val_losses,Accuracy=fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[AccumulatedAccuracyMetric()])

PATH = './CNN_DED.pth'
torch.save(model.state_dict(), PATH)
torch.save(model, PATH)

train_plot = 'CNN_train_losses'+'_'+ '.npy'
val_plot = 'CNN_val_losses'+'_'+'.npy'
Accuracy_plot = 'CNN_Accuracy_losses'+'_'+'.npy'

np.save(train_plot,train_losses, allow_pickle=True)
np.save(val_plot,val_losses, allow_pickle=True)
np.save(Accuracy_plot,Accuracy, allow_pickle=True)

#%% Plot training curves

plot_curves(train_losses,val_losses,Accuracy)

#%%
train_embeddings = 'CNN_train_embeddings'+ '.npy'
train_labelsname = 'CNN_train_labels'+'.npy'
test_embeddings = 'CNN_test_embeddings'+ '.npy'
test_labelsname = 'CNN_test_labels'+'.npy'
#%%
X_train, y_train = extract_embeddings(train_loader, model,train_embeddings,train_labelsname)
X_test, y_test = extract_embeddings(test_loader, model,test_embeddings,test_labelsname)

#%% Embedding plots/ Dimensional reduction plots

_,_,tsne=TSNEplot(X_train,y_train,perplexity=50)
test_embeddings,test_labels=TSNEtransform(tsne,X_test,y_test)
   

graph_name='CNN_TSNE_Latent' +'_'+'.png'
ax,fig=Three_embeddings(test_embeddings,test_labels,graph_name,ang=195)#135
gif_name= str('CNN_TSNE_Latent')+'.gif'

def rotate(angle):
      ax.view_init(azim=angle)
      
angle = 3
ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
ani.save(gif_name, writer=animation.PillowWriter(fps=20))


graph_name_2D='CNN_Training_Feature_2D' +'_'+'.png'
graph_title = "Feature space distribution"
plot_embeddings(X_test,y_test,classes,graph_title,graph_name_2D, xlim=None, ylim=None)


graph_name_3D='CNN_Latent_3D' +'_'+'.png'
ax,fig=Three_Latent_embeddings(X_train,y_train,graph_name_3D,ang=195, dim_1=20, dim_2=30, dim_3=31)#135
gif2_name= str('CNN_3D_Latent')+'.gif'
angle = 3
ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
ani.save(gif2_name, writer=animation.PillowWriter(fps=20))


Distribution_name_2D='CNN_Distribution_2D_'
distribution_plots(X_train,y_train,Distribution_name_2D)


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
  Cummulative_plots(X_train,y_train,i,ax)
  
fig.tight_layout();
fig.savefig(graph_name_32D)
fig.show()
# fig.clf()

#%%Accuracy on the classifier

classifier(X_train, X_test, y_train, y_test,'LogisticRegression','CNN')
classifier(X_train, X_test, y_train, y_test,'SVM','CNN')
classifier(X_train, X_test, y_train, y_test,'RF','CNN')
classifier(X_train, X_test, y_train, y_test,'GaussianNB','CNN')
