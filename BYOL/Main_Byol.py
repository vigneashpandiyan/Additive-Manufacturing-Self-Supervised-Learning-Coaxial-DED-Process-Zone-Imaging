# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 11:47:42 2022
@author: srpv


"""
import numpy as np
import os
import matplotlib.pyplot as plt

from torchvision import transforms
import torch
import torchvision.transforms as transforms 
import torchvision 
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
from matplotlib import animation
import math
from Model.losses import *
from Model.Networks import *
from Model.Augmentations import *
from Model.Byoltrainer import *
from Utils.Utils import *
from sklearn.preprocessing import StandardScaler

from MLClassification.Classifier import *
from Anomaly.Utils_one_class_SVM import *

cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Setting Up CUDA
print(torch.cuda.is_available())


#%%
# Data--> https://polybox.ethz.ch/index.php/s/HUcJ7cJ18K0MrEn 
# datadir = '../Data/train'   #place data inside 
#           '../Data/test'  

datadir ="C:/Users/srpv/Desktop/DED Byol/Data/"  #path towards the dataset
traindir = datadir + 'train/'
testdir = datadir + 'test/'

categories = [[folder, os.listdir(traindir + folder)] for folder in os.listdir(traindir)  if not folder.startswith('.') ]

epoch=100
batch_size=256
#%% Dataset preparation / Data loader

data_transform = transforms.Compose([
        torchvision.transforms.Resize((480,320)),
        transforms.ToTensor()])
    
train_dataset = datasets.ImageFolder(root=traindir,transform=data_transform)
test_dataset = datasets.ImageFolder(root=testdir,transform=data_transform)
classes = train_dataset.classes


train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=False,)

num_step = math.floor(len(train_loader.dataset) / batch_size)


unlabeled_loader = DataLoader(test_dataset,batch_size=512,shuffle=True,drop_last=False,)

#%% Model
embeddinglayer_size=64
model = EmbeddingNet(dropout=0.05,embedding=embeddinglayer_size)
model = BYOL(model, augmentation_tau(), augmentation_tau_prime(), #Augmentation to the dataset
                embedding = embeddinglayer_size,epoch=epoch,num_step=num_step,projection_size=32).cuda()

count_parameters(model)
history = model.fit(train_loader,unlabeled_loader)
plot_curves(history)

#%%

train_plot = 'train_losses'+'_'+ '.npy'
val_plot = 'val_losses'+'_'+'.npy'

np.save(train_plot,history["train_loss"], allow_pickle=True)
np.save(val_plot,history["val_loss"], allow_pickle=True)

PATH = './Byol_DED.pth'
torch.save(model.state_dict(), PATH)
torch.save(model, PATH)


#%%
train_embeddings = 'Byol_train_embeddings'+ '.npy'
train_labelsname = 'Byol_train_labels'+'.npy'
test_embeddings = 'Byol_test_embeddings'+ '.npy'
test_labelsname = 'Byol_test_labels'+'.npy'

X_train, y_train = extract_embeddings(train_loader, model,train_embeddings,train_labelsname)
X_test, y_test = extract_embeddings(unlabeled_loader, model,test_embeddings,test_labelsname)

standard_scaler = StandardScaler()
X_train=standard_scaler.fit_transform(X_train)
X_test=standard_scaler.fit_transform(X_test)

#%% Embedding plots/ Dimensional reduction plots

_,_,tsne=TSNEplot(X_train,y_train,perplexity=10)

test_embeddings,test_labels=TSNEtransform(tsne,X_test,y_test)
   

graph_name='Byol_TSNE_Latent' +'_'+'.png'
ax,fig=Three_embeddings(test_embeddings,test_labels,graph_name,ang=195)#135
gif_name= str('Byol_TSNE_Latent')+'.gif'


def rotate(angle):
      ax.view_init(azim=angle)
      
angle = 3
ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
ani.save(gif_name, writer=animation.PillowWriter(fps=20))


graph_name_2D='Byol_Training_Feature_2D' +'_'+'.png'
graph_title = "Feature space distribution"
plot_embeddings(X_test,y_test,classes,graph_title,graph_name_2D, xlim=None, ylim=None)


graph_name_3D='Byol_Latent_3D' +'_'+'.png'
ax,fig=Three_Latent_embeddings(X_train,y_train,graph_name_3D,ang=170, dim_1=20, dim_2=30, dim_3=31)#135


gif2_name= str('Byol_3D_Latent')+'.gif'
angle = 3
ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
ani.save(gif2_name, writer=animation.PillowWriter(fps=20))


Distribution_name_2D='Distribution_2D_'
distribution_plots(X_train,y_train,Distribution_name_2D)


fig, axs = plt.subplots(
  nrows=8,
  ncols=4,
  sharey=False,
  figsize=(11, 14),
  dpi=1000
)

columns = np.atleast_2d(X_train).shape[1]
graph_name_32D='Byol_Latent_32D' +'_'+'.png'
for i in range(columns):
  ax = axs.flat[i]
  Cummulative_plots(X_train,y_train,i,ax)
  
fig.tight_layout();
fig.savefig(graph_name_32D)
fig.show()


#%%Accuracy on the classifier

classifier(X_train, X_test, y_train, y_test,'LogisticRegression','Byol')
classifier(X_train, X_test, y_train, y_test,'SVM','Byol')
classifier(X_train, X_test, y_train, y_test,'RF','Byol')
classifier(X_train, X_test, y_train, y_test,'GaussianNB','Byol')


#%%

train_embeddings = 'Byol_train_embeddings'+ '.npy'
X_train=np.load(train_embeddings)
train_labelsname = 'Byol_train_labels'+'.npy'
y_train=np.load(train_labelsname)
test_embeddings = 'Byol_test_embeddings'+ '.npy'
X_test=np.load(test_embeddings)
test_labelsname = 'Byol_test_labels'+'.npy'
y_test=np.load(test_labelsname)

anomaly_detection(X_train,X_test,y_train,y_test)



