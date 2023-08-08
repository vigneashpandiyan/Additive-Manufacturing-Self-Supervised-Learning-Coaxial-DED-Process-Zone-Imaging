
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 11:47:42 2022
@author: srpv

"""

import pandas as pd
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable
from sklearn.manifold import TSNE
cuda = torch.cuda.is_available()
import os
from sklearn.preprocessing import StandardScaler
#%% 

'''
Compute the total number of trainable parameters
... in the model
'''
     
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

#%%

'''
Plot the training curves...
'''

def plot_curves(history):
    
    plt.rcParams.update(plt.rcParamsDefault)
    
    fig = plt.figure(figsize=(6,4), dpi=100)
    
    plt.xlabel('Epoch',fontsize = 17)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.ylabel('Loss values',fontsize = 17)
    plt.plot(history["train_loss"],  marker='o',c='red', label='Training Loss',linewidth =1.8)
    plt.plot(history["val_loss"],marker='*', c='g', label="Validation Loss",linewidth =1.8)
    plt.legend( loc='upper right',fontsize = 20)
    plt.savefig('Byol_Loss.png', dpi=600,bbox_inches='tight')
    plt.show()
    
    
#%%

'''
Compute the embeddings from the network
... in the model/ save the embeddings

'''

def extract_embeddings(dataloader, model,embeddings_name,labels_name):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 32))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            images1 = images
            target=target
            if cuda:
                images = images.cuda()
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
            
            
    embeddings=embeddings.astype(np.float64)
    labels=labels.astype(np.float64)
    np.save(embeddings_name,embeddings, allow_pickle=True)
    np.save(labels_name,labels, allow_pickle=True)
    
    return embeddings, labels

#%%

'''
Compute the t-sne on the embeddings from the network
'''
def TSNEplot(output,target,perplexity):
    
    
    print('target shape: ', target.shape)
    print('output shape: ', output.shape)
    print('perplexity: ',perplexity)
    

    group=target
    group = np.ravel(group)
        
    RS=np.random.seed(123)
    tsne = TSNE(n_components=3, random_state=RS, perplexity=perplexity)
    tsne_fit = tsne.fit_transform(output)
        
    return tsne_fit,target,tsne

def TSNEtransform(tsne,data,test_labels):
    tsne_fit = tsne.fit_transform(data)
    target = test_labels
    return tsne_fit,target


#%%


'''
Plot t-sne embedding in 3D
'''

def Three_embeddings(embeddings, targets,graph_name,ang, xlim=None, ylim=None):
    
    folder = os.path.join('Figures/', 'BYOL')
    
    try:
        os.makedirs(folder, exist_ok = True)
        print("Directory created....")
    except OSError as error:
        print("Directory already exists....")
    
    print(folder)
    folder=folder+'/'
    
    
    df2 = pd.DataFrame(targets) 
    df2.columns = ['Categorical']
    df2=df2['Categorical'].replace(0,'P1')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(1,'P2')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(2,'P3')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(3,'P4')
    df2 = pd.DataFrame(df2) 
    targets = pd.DataFrame(df2) 
    
    targets=targets.to_numpy()
    targets = np.ravel(targets)
    
    
    x1=embeddings[:, 0]
    x2=embeddings[:, 1]
    x3=embeddings[:, 2]
    
    
    df = pd.DataFrame(dict(x=x1, y=x2,z=x3, label=targets))
    groups = df.groupby('label')
    uniq = list(set(df['label']))
    uniq=np.sort(uniq)
    
    
    plt.rcParams.update(plt.rcParamsDefault)
    
    fig = plt.figure(figsize=(12,6), dpi=100)
    fig.set_facecolor('white')
    plt.rcParams["legend.markerscale"] = 2
   
    ax = plt.axes(projection='3d')
    
    ax.grid(False)
    ax.view_init(azim=ang)#115
    
    color = [ '#0000FF','orange','green','red', 'blue', 'cyan']
    marker= ["*",">","X","o","d","s",]
    
    
    ax.set_facecolor('white') 
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False
    
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    
    
    ax.tick_params(axis='both', labelsize=20)
    
    
    graph_title = "Feature space distribution"
    
    j=0
    for i in uniq:
        print(i)
        indx = targets == i
        a=x1[indx]
        b=x2[indx]
        c=x3[indx]
        ax.plot(a, b, c ,color=color[j],label=uniq[j],marker=marker[j],linestyle='',ms=7)
        j=j+1
     
    plt.xlabel ('Dimension-1', labelpad=10)
    plt.ylabel ('Dimension-2', labelpad=10)
    ax.set_zlabel('Dimension-3',labelpad=10)
    
    plt.title(str(graph_title))
    plt.legend(loc='upper left',frameon=False)
    plt.savefig(os.path.join(folder, graph_name), bbox_inches='tight',dpi=400)
    plt.show()
    
    return ax,fig

#%%


'''
Plot embedding in 2D
'''

def plot_embeddings(embeddings, targets,classes,graph_title,graph_name_2D, xlim=None, ylim=None):
    
    standard_scaler = StandardScaler()
    embeddings=standard_scaler.fit_transform(embeddings)
    
    folder = os.path.join('Figures/', 'BYOL')
    
    try:
        os.makedirs(folder, exist_ok = True)
        print("Directory created....")
    except OSError as error:
        print("Directory already exists....")
    
    print(folder)
    folder=folder+'/'
    
    color = [ '#0000FF','orange','green','red', 'blue', 'cyan']
    marker= ["*",">","X","o","d","s",]
    mnist_classes = ['P1', 'P2', 'P3', 'P4','P5','P6']
    plt.rcParams.update(plt.rcParamsDefault)
    plt.figure(figsize=(7,5))
    j=0
    for i in range(len(classes)):
        print(i)
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,16], embeddings[inds,17], alpha=0.7, color=color[j],marker=marker[j],s=100)
        j=j+1
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(mnist_classes,bbox_to_anchor=(1.16, 1.05))
    plt.xlabel ('Weights_1', labelpad=10)
    plt.ylabel ('Weights_2', labelpad=10)
    plt.title(str(graph_title),fontsize = 15)
    
    
    plt.savefig(os.path.join(folder, graph_name_2D), bbox_inches='tight',dpi=600)
    plt.show()


#%%

'''
Plot embedding in 3D / based on 3 latent vectors
'''

def Three_Latent_embeddings(embeddings, targets,graph_name,ang, dim_1, dim_2, dim_3,xlim=None, ylim=None):
    
    folder = os.path.join('Figures/', 'BYOL')
    
    try:
        os.makedirs(folder, exist_ok = True)
        print("Directory created....")
    except OSError as error:
        print("Directory already exists....")
    
    print(folder)
    folder=folder+'/'
     
    df2 = pd.DataFrame(targets) 
    df2.columns = ['Categorical']
    df2=df2['Categorical'].replace(0,'P1')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(1,'P2')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(2,'P3')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(3,'P4')
    df2 = pd.DataFrame(df2) 
    targets = pd.DataFrame(df2) 
    
    targets=targets.to_numpy()
    targets = np.ravel(targets)
    
    x1=embeddings[:, dim_1]
    x2=embeddings[:, dim_2]
    x3=embeddings[:, dim_3]
    
    
    df = pd.DataFrame(dict(x=x1, y=x2,z=x3, label=targets))
    groups = df.groupby('label')
    uniq = list(set(df['label']))
    uniq=np.sort(uniq)
    #uniq=["0","1","2","3"]
    
    plt.rcParams.update(plt.rcParamsDefault)
    
    fig = plt.figure(figsize=(12,6), dpi=100)
    fig.set_facecolor('white')
    plt.rcParams["legend.markerscale"] = 2
   
    ax = plt.axes(projection='3d')
    
    ax.grid(False)
    ax.view_init(azim=ang)#115
    
    color = [ '#0000FF','orange','green','red', 'blue', 'cyan']
    marker= ["*",">","X","o","d","s",]
    
    
    ax.set_facecolor('white') 
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False
    
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    
    
    ax.tick_params(axis='both', labelsize=20)
    
    
    graph_title = "Feature space distribution"
    
    j=0
    for i in uniq:
        print(i)
        indx = targets == i
        a=x1[indx]
        b=x2[indx]
        c=x3[indx]
        ax.plot(a, b, c ,color=color[j],label=uniq[j],marker=marker[j],linestyle='',ms=7)
        j=j+1
     
    plt.xlabel ('Dimension-1', labelpad=10)
    plt.ylabel ('Dimension-2', labelpad=10)
    ax.set_zlabel('Dimension-3',labelpad=10)
    plt.title(str(graph_title))
    plt.legend(loc='upper left',frameon=False)
    plt.savefig(os.path.join(folder, graph_name), bbox_inches='tight',dpi=400)
    plt.show()
    return ax,fig




#%%
def dist_plot(data,i,Net):
    
    
    folder = os.path.join('Figures/', 'BYOL')
    
    try:
        os.makedirs(folder, exist_ok = True)
        print("Directory created....")
    except OSError as error:
        print("Directory already exists....")
    
    print(folder)
    folder=folder+'/'
    
    new_columns = list(data.columns)
    new_columns[-1] = 'target'
    data.columns = new_columns
    data.target.value_counts()
    data = data.sample(frac=1.0)
    
    data_1 = data[data.target == 'P1']
    data_1 = data_1.drop(labels='target', axis=1)
    data_2 = data[data.target == 'P2']
    data_2 = data_2.drop(labels='target', axis=1)
    data_3 = data[data.target == 'P3']
    data_3 = data_3.drop(labels='target', axis=1)
    data_4 = data[data.target == 'P4']
    data_4 = data_4.drop(labels='target', axis=1)
    
    
    plt.rcParams.update(plt.rcParamsDefault)
    
    sns.set(style="white")
    fig=plt.subplots(figsize=(5,3), dpi=800)
    fig = sns.kdeplot(data_1['Feature'], shade=True,alpha=.5, color="#0000FF")
    fig = sns.kdeplot(data_2['Feature'], shade=True,alpha=.5, color="Orange")
    fig = sns.kdeplot(data_3['Feature'], shade=True,alpha=.5, color="green")
    fig = sns.kdeplot(data_4['Feature'], shade=True,alpha=.5, color="red")
    
    
    data=pd.concat([data_1,data_2,data_3],axis=1) 
    data=data.to_numpy()
    
    plt.title("Weight " + str(i))
    plt.legend(labels=['P1','P2','P3','P4'],bbox_to_anchor=(1.24, 1.05))
    title=str(Net)+'_'+str(i)+'_'+'distribution_plot'+'.png'
    # plt.xlim([0.0, np.max(data)])
    # plt.ylim([0.0, 65])
    plt.xlabel('Weight distribution') 
    plt.savefig(os.path.join(folder, title), bbox_inches='tight')
    plt.show()
    

def distribution_plots(Featurespace,classspace,Net):
    
    columns = np.atleast_2d(Featurespace).shape[1]
    df2 = pd.DataFrame(classspace)
    
    df2.columns = ['Categorical']
    df2=df2['Categorical'].replace(0,'P1')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(1,'P2')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(2,'P3')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(3,'P4')
    df2 = pd.DataFrame(df2) 
    
    print(columns)
    
    for i in range(columns):
        print(i)
        Featurespace_1 = Featurespace.transpose()
        data=(Featurespace_1[i])
        data=data.astype(np.float64)
        #data= abs(data)
        df1 = pd.DataFrame(data)
        df1.rename(columns={df1.columns[0]: "Feature" }, inplace = True)
        df2.rename(columns={df2.columns[0]: "categorical" }, inplace = True)
        data = pd.concat([df1, df2], axis=1)
        
        minval = min(data.categorical.value_counts())
        data = pd.concat([data[data.categorical == cat].head(minval) for cat in data.categorical.unique() ])
        
        dist_plot(data,i,Net)
        
#%%

def Cummulative_plots(Featurespace,classspace,i,ax):
    
    columns = np.atleast_2d(Featurespace).shape[1]
    df2 = pd.DataFrame(classspace)
    
    df2.columns = ['Categorical']
    df2=df2['Categorical'].replace(0,'P1')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(1,'P2')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(2,'P3')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(3,'P4')
    df2 = pd.DataFrame(df2) 
    
    
    print(i)
    
    Featurespace_1 = Featurespace.transpose()
    data=(Featurespace_1[i])
    data=data.astype(np.float64)
    
    df1 = pd.DataFrame(data)
    df1.rename(columns={df1.columns[0]: "Feature" }, inplace = True)
    df2.rename(columns={df2.columns[0]: "categorical" }, inplace = True)
    data = pd.concat([df1, df2], axis=1)
    minval = min(data.categorical.value_counts())
    data = pd.concat([data[data.categorical == cat].head(minval) for cat in data.categorical.unique() ])
    
    Cummulative_dist_plot(data,i,ax)
    
def Cummulative_dist_plot(data,i,ax):
    new_columns = list(data.columns)
    new_columns[-1] = 'target'
    data.columns = new_columns
    data.target.value_counts()
    data = data.sample(frac=1.0)
    
    data_1 = data[data.target == 'P1']
    data_1 = data_1.drop(labels='target', axis=1)
    data_2 = data[data.target == 'P2']
    data_2 = data_2.drop(labels='target', axis=1)
    data_3 = data[data.target == 'P3']
    data_3 = data_3.drop(labels='target', axis=1)
    data_4 = data[data.target == 'P4']
    data_4 = data_4.drop(labels='target', axis=1)
    
    plt.rcParams.update(plt.rcParamsDefault)
    sns.set(style="white")
    
    ax.plot(figsize=(5,5), dpi=800)
    sns.kdeplot(data_1['Feature'], shade=True,alpha=.5, color="#0000FF",ax=ax)
    sns.kdeplot(data_2['Feature'], shade=True,alpha=.5, color="Orange",ax=ax)
    sns.kdeplot(data_3['Feature'], shade=True,alpha=.5, color="green",ax=ax)
    sns.kdeplot(data_4['Feature'], shade=True,alpha=.5, color="red",ax=ax)
    
    ax.set_title("Weight " + str(i), y=1.0, pad=-14)
    ax.set_xlabel('Weight distribution') 
    # ax.set_ylabel('Density')
    
    ax.tick_params(axis='both', labelsize=20)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    

    
#%%




