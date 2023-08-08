
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from prettytable import PrettyTable
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
import os
#%%

def plot_confusion_matrix(y_true, y_pred,classes,plotname):
            
    # Build confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Normalise
    cmn = cm.astype('float')  / cm.sum(axis=1)[:, np.newaxis]
    cmn=cmn*100
    
    fig, ax = plt.subplots(figsize=(12,9))
    sns.set(font_scale=3) 
    b=sns.heatmap(cmn, annot=True, fmt='.1f', xticklabels=classes, yticklabels=classes,cmap="coolwarm",linewidths=0.1,annot_kws={"size": 25},cbar_kws={'label': 'Classification Accuracy %'})
    for b in ax.texts: b.set_text(b.get_text() + " %")
    plt.ylabel('Actual',fontsize=25)
    plt.xlabel('Predicted',fontsize=25)
    plt.margins(0.2)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=90, va="center", fontsize= 20)
    ax.set_xticklabels(ax.get_xticklabels(), va="center",fontsize= 20)
    # plt.setp(ax.get_yticklabels(), rotation='vertical')
    plotname=str(plotname)
    plt.savefig(plotname,bbox_inches='tight')
    plt.show()
    plt.clf()

#%%

def plots(train_loss_history,valid_loss_history,Learning_rate):
    
    
    Accuracyfile = 'Accuracy'+'.npy'
    Lossfile = 'Loss_value'+'.npy'

    np.save(Accuracyfile,train_loss_history,allow_pickle=True)
    np.save(Lossfile,valid_loss_history, allow_pickle=True)
    
    
    fig, ax = plt.subplots(figsize = (6, 4),dpi=200)
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    plt.plot(train_loss_history, marker='o',c='red', label='Training Loss',linewidth =1.8)
    plt.plot(valid_loss_history,marker='*', c='g', label="Validation Loss",linewidth =1.8)
    plt.xlabel('Epoch',fontsize = 16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel('Loss values',fontsize = 16)
    plt.legend()
    plt.savefig("Training.png",bbox_inches='tight',dpi=200)
    plt.show()
    
    
    
    fig, ax = plt.subplots(figsize = (7, 4),dpi=200)
    plt.plot(Learning_rate,'green',linewidth =2.0)
    plt.title('Learning_Rate')
    plt.xlabel('Epochs')
    plt.ylabel('Learning_Rate')
    plot_3=  'Learning_rate_'+ '.png'
    plt.savefig(plot_3, dpi=600,bbox_inches='tight')
    plt.show()


#%%
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
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.7, color=color[j],marker=marker[j],s=100)
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
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    

def Compute_latents(trained_model,loader,device,name):


    y_pred = []
    y_true = []
    
    embeddings = np.zeros((len(loader.dataset), 32))
    labels = np.zeros(len(loader.dataset))
    trained_model.eval()
    k = 0
    for data, target in loader:
         
         data,target = data.to(device,dtype=torch.float),target.to(device,dtype=torch.long)
         target=target.squeeze()
         
         output, attn_matrix,latent = trained_model(data, return_attn_matrix = True)
         embeddings[k:k+len(data)]=latent.cpu().detach().numpy()
         labels[k:k+len(data)] = target.cpu().detach().numpy()
         k += len(data)
         
         output = torch.argmax(output, dim=1) 
         output=output.data.cpu().numpy()
         target=target.data.cpu().numpy()
         y_true.extend(output) # Save Truth 
         y_pred.extend(target) # Save Prediction
     
     
    embeddings=embeddings.astype(np.float64)
    labels=labels.astype(np.float64)

    train_embeddings = 'CNN_embeddings_'+str(name) +'.npy'
    train_labelsname = 'CNN_labels_'+str(name) +'.npy'

    np.save(train_embeddings,embeddings, allow_pickle=True)
    np.save(train_labelsname,labels, allow_pickle=True)
     
    return y_true, y_pred,embeddings, labels