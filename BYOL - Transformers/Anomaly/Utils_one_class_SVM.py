# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 13:37:13 2022

@author: srpv
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
import pandas as pd
import os
from sklearn.model_selection import train_test_split# implementing train-test-split
from sklearn import svm
from sklearn.manifold import TSNE


def anomaly_detection(X_train,X_test,y_train,y_test):
    
   
    Featurespace=np.concatenate((X_train,X_test))
    classspace=np.concatenate((y_train,y_test))

    
    X_Features,X_labels,Y_Features,Y_labels = anomaly_data_prep(Featurespace, classspace)
    anomaly_one_class_SVM(X_Features,X_labels,Y_Features,Y_labels)


def anomaly_one_class_SVM(X_Features,X_labels,Y_Features,Y_labels):
    plt.rcParams.update(plt.rcParamsDefault)
    folder = os.path.join('Figures/', 'Anomaly')
    
    try:
        os.makedirs(folder, exist_ok = True)
        print("Directory created....")
    except OSError as error:
        print("Directory already exists....")
    
    print(folder)
    folder=folder+'/'
    
    
    Outliers=Y_Features.to_numpy()
    X_Features=X_Features.to_numpy()
    
    X_train, X_test, y_train, y_test = train_test_split(X_Features, X_labels, test_size=0.25, random_state=66)
    
    

    # fit the model
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma='auto')
    clf.fit(X_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    y_pred_outliers = clf.predict(Outliers)
    n_error_train = y_pred_train[y_pred_train == -1].size
    n_error_test = y_pred_test[y_pred_test == -1].size
    n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size
    
    
    print(
        "error train: %d/%d ; errors novel regular: %d/%d ; "
        "errors novel abnormal: %d/%d"
        % (n_error_train,len(X_train), n_error_test,len(X_test), n_error_outliers,len(Outliers)))

    
    
    fig = plt.figure(figsize=(9,9), dpi=800)
    plt.rcParams["legend.markerscale"] = 3
    plt.rcParams['xtick.labelsize']=25
    plt.rcParams['ytick.labelsize']=25
    
    s = 60
    c = plt.scatter(Outliers[:, 0], Outliers[:, 1], c="orange", s=s, edgecolors="k",alpha=0.2)
    b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c="cyan", s=s, edgecolors="k")
    b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c="lightblue", s=s, edgecolors="k")
    
    plt.axis("tight")
    plt.xlabel ('Dimension 1', labelpad=5,fontsize=30)
    plt.ylabel ('Dimension 2', labelpad=5,fontsize=30)
    
    # plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.legend(
        [ b1, b2, c],
        [
            
            "Training (conduction)",
            "Testing (conduction)",
            "Outlier observations (LoF,Keyhole)",
        ],
        loc="upper right",
        bbox_to_anchor=(0.90, -0.15),
        prop=matplotlib.font_manager.FontProperties(size=25),
    )
    
    plotname=folder+'anomaly_classification_accuracy.png'
    plt.savefig(plotname, bbox_inches='tight',dpi=800)
    plt.show()



def anomaly_data_prep(X, y):
    
    
    y = np.ravel(y)
    RS=np.random.seed(1974)
    perplexity=20
    tsne = TSNE(n_components=3, random_state=RS, perplexity=perplexity)
    X = tsne.fit_transform(X)
   
    print(X.shape)
    X = pd.DataFrame(X) 
    labels = pd.DataFrame(y) 
    
    labels.columns = ['Categorical']
    labels=labels['Categorical'].replace(0,-1)
    labels = pd.DataFrame(labels) 
    labels=labels['Categorical'].replace(2,-1)
    labels = pd.DataFrame(labels)
    labels=labels['Categorical'].replace(3,-1)
    labels = pd.DataFrame(labels)
    
    df=pd.concat([X,labels], axis=1)
    new_columns = list(df.columns)
    new_columns[-1] = 'target'
    df.columns = new_columns
    df.target.value_counts()
    df = df.sample(frac=1.0)
    
    
    df_1 = df[df.target == 1]
    print(df_1.shape)
    
    df_2 = df[df.target != 1]
    print(df_2.shape)
    
    
    X_labels=df_1.iloc[:,-1]
    X_labels = pd.DataFrame(X_labels) 
    X_Features = df_1.drop(labels='target', axis=1)
    
    Y_labels=df_2.iloc[:,-1]
    Y_labels = pd.DataFrame(Y_labels) 
    Y_Features = df_2.drop(labels='target', axis=1)
    
    
    
    return X_Features,X_labels,Y_Features,Y_labels
    
    
    
