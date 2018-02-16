# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 18:32:09 2018

@author: ChenZhengyang
"""

import os
import numpy as np 
import pandas as pd 
from sklearn import preprocessing
from sklearn import metrics
from sklearn import tree
from sklearn.externals.six import StringIO
import pydotplus as pdp
import sklearn.metrics as mat

dataset = pd.read_csv("../DATASET/student/student-por.csv")

#observe the types of data
dataset.dtypes

#observe distributions of object-type features
var = ['school','sex','address','famsize','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup',
       'paid','activities','nursery','higher','internet','romantic']
"""for v in var:
    print('\nFrequency count for variable %s'%v) 
    print(dataset[v].value_counts())
    """
    
#label encode 
from sklearn.preprocessing import LabelEncoder
var_to_encode = ['school','sex','address','famsize','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup',
       'paid','activities','nursery','higher','internet','romantic']
for col in var_to_encode:
    dataset[col] = LabelEncoder().fit_transform(dataset[col])

# Binarize G3<=11: G3=0   G3>11: G3=1
dataset[['G3']] = preprocessing.Binarizer(threshold=11).transform(dataset[['G3']])

x=dataset[dataset.columns.drop('G3')]
y= dataset['G3']

# divide dataset into train set and test set, size of test equals = 0.33*size of dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
     x,y,test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(x_train)  
x_train = scaler.transform(x_train)  
# apply same transformation to test data
x_test = scaler.transform(x_test)  

#--------------------------------------END OF PREPROCESSING----------------------

from sklearn import neural_network as nw

itermax=50
nw_auc_trn1=np.zeros(itermax)
nw_auc_tst1=np.zeros(itermax)
for i in range(3,itermax):
    clf1= nw.MLPClassifier(solver='lbfgs',max_iter=i,alpha=1e-5,hidden_layer_sizes=(50), random_state=1)
    clf1 = clf1.fit(x_train, y_train)
    nw_auc_tst1[i] = mat.roc_auc_score(y_test, clf1.predict_proba(x_test)[:,1])
    nw_auc_trn1[i] = mat.roc_auc_score(y_train, clf1.predict_proba(x_train)[:,1])
    
itermax=50
nw_auc_trn2=np.zeros(itermax)
nw_auc_tst2=np.zeros(itermax)
for i in range(3,itermax):
    clf2= nw.MLPClassifier(solver='lbfgs',max_iter=i,alpha=1e-5,hidden_layer_sizes=(50, 50), random_state=1)
    clf2 = clf2.fit(x_train, y_train)
    nw_auc_tst2[i] = mat.roc_auc_score(y_test, clf2.predict_proba(x_test)[:,1])
    nw_auc_trn2[i] = mat.roc_auc_score(y_train, clf2.predict_proba(x_train)[:,1])

#print (nw_auc_tst1,nw_auc_trn1,nw_auc_tst2,nw_auc_trn2)

from matplotlib import pyplot
pyplot.plot(nw_auc_tst1, linewidth=2, label = "1 HiddenlayerNeural test AUC")
pyplot.plot(nw_auc_trn1, linewidth=2, label = "1 HiddenlayerNeural train AUC")
pyplot.plot(nw_auc_tst2, linewidth=2, label = "2 Hiddenlayer Neural test AUC")
pyplot.plot(nw_auc_trn2, linewidth=2, label = "2 Hiddenlayer Neural train AUC")
pyplot.legend()
pyplot.xlim(0, 50)
pyplot.ylim(0.8, 1.0)
pyplot.xlabel("Max_ITERATION")
pyplot.ylabel("AUC SCORE")
#plt.figure(figsize=(12,12))
pyplot.show()


act_clfscore=[]
acts=['identity','logistic','tanh','relu']
for act in acts:
    clf=nw.MLPClassifier(activation=act,solver='lbfgs',hidden_layer_sizes=(30,30),random_state=1)
    clf=clf.fit(x_train,y_train)
    act_clfscore.append(mat.roc_auc_score(y_test,clf.predict_proba(x_test)[:,1]))

ind = np.arange(4)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = pyplot.bar(ind, act_clfscore, width, color='#d62728')

pyplot.ylabel('Scores')
pyplot.title('Scores by different activations')
pyplot.xticks(ind, acts)
pyplot.yticks(np.arange(0, 1.01, 0.1))
pyplot.ylim(0.8, 1.0)

pyplot.show()






