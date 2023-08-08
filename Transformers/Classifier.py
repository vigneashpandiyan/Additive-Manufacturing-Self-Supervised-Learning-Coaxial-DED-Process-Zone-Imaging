
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 11:47:42 2022
@author: srpv

"""

from sklearn.metrics import classification_report,confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import RepeatedStratifiedKFold
import joblib
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import pandas as pd
import joblib
from sklearn.model_selection import cross_val_score
from IPython.display import Image
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split# implementing train-test-split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import os
#%%


def Dataframe_Manipulation(Distance,target):
        
       
    df1 = pd.DataFrame(Distance) 
    df1.columns = ['Distance']
    df2 = pd.DataFrame(target) 
    df2.columns = ['Categorical']
    
    df2=df2['Categorical'].replace(0,'P1')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(1,'P2')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(2,'P3')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(3,'P4')
    df2 = pd.DataFrame(df2) 
     
    
    df=pd.concat([df1,df2], axis=1)
    new_columns = list(df.columns)
    new_columns[-1] = 'Target'
    df.columns = new_columns
    df.Target.value_counts()
    df = df.sample(frac=1.0)
    
    print(df.shape)
    
    return df


def Dataframe_Manipulation_Classifier(target):
        
       
    
    df2 = pd.DataFrame(target) 
    df2.columns = ['Categorical']
    
    df2=df2['Categorical'].replace(0,'P1')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(1,'P2')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(2,'P3')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(3,'P4')
    df2 = pd.DataFrame(df2) 
    
    
    return df2

#%%

def classifier(X_train, X_test, y_train, y_test,modelname,CNN):
    
    
    folder = os.path.join('Figures/', 'MLclassifier')
    
    try:
        os.makedirs(folder, exist_ok = True)
        print("Directory created....")
    except OSError as error:
        print("Directory already exists....")
    
    print(folder)
    folder=folder+'/'

    
    y_train=Dataframe_Manipulation_Classifier(y_train)
    y_test=Dataframe_Manipulation_Classifier(y_test)

    if modelname == 'LogisticRegression':
        model = LogisticRegression(max_iter=1000, random_state=123)
        print("this will do the calculation")
    elif modelname == 'SVM':
        model = SVC(kernel='rbf', probability=True)
    elif modelname == 'RF':
        model = RandomForestClassifier(n_estimators=100 , oob_score=True)
    elif modelname == 'GaussianNB':
        model = GaussianNB()
    else:
        exit()
    
       
    model.fit(X_train,y_train)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model,X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1)
    print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))  
    predictions = model.predict(X_test)
    
    print(str(modelname))
    print(metrics.accuracy_score(y_test, predictions))
    print(classification_report(y_test,predictions))
    print(confusion_matrix(y_test,predictions))
    
    
    
    graph_name1= str(CNN)+'_'+str(modelname)+'_without normalization w/o Opt'
    graph_name2= str(CNN)+'_'+str(modelname)
    
    graph_1= str(CNN)+'_'+str(modelname)+'_Confusion_Matrix'+'_'+'No_Opt'+'.png'
    graph_2= str(CNN)+'_'+str(modelname)+'_Confusion_Matrix'+'_'+'Opt'+'.png'
    
    
    titles_options = [(graph_name1, None, graph_1),
                      (graph_name2, 'true', graph_2)]
    
    for title, normalize ,graphname  in titles_options:
        plt.figure(figsize = (20, 10),dpi=400)
        disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test,
                                      display_labels=['P1','P2','P3','P4'],
                                      cmap=plt.cm.Reds,xticks_rotation='vertical',
                                    normalize=normalize,values_format = '.2f')
        plt.title(title, size = 12)
        
        plt.savefig(os.path.join(folder, graphname),bbox_inches='tight',dpi=400)
    savemodel=  folder+str(CNN)+'_'+str(modelname)+'_LR'+'_model'+'.sav'    
    joblib.dump(model, savemodel)
    
#%%



