# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 12:50:44 2018

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
from matplotlib import pyplot


train = pd.read_csv("../DATASET/BANK/MT_Train.csv")
test = pd.read_csv('../DATASET/BANK/MT_Test.csv')

#
train['source']= 'train'
test['source'] = 'test'
data=pd.concat([train, test],ignore_index=True)
#data.shape

#
data.drop('default',axis=1,inplace=True)
data.drop('emp.var.rate',axis=1,inplace=True)

#
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
var_to_encode = ['job','marital','education','day_of_week','month','housing','loan','poutcome']
for col in var_to_encode:
    data[col] = le.fit_transform(data[col])
    
#
#data = pd.get_dummies(data, columns=var_to_encode)
#data.columns
#
data["contact"]=preprocessing.LabelBinarizer().fit_transform(data["contact"])
#
data[["pdays"]] = preprocessing.Binarizer(threshold=998).transform(data[["pdays"]])


#
train_mod = data.loc[data['source']=='train']
test_mod = data.loc[data['source']=='test']
train=train_mod.copy()
test=test_mod.copy()
train.drop(['source','SampleId'],axis=1,inplace=True)
test.drop(['source','y'],axis=1,inplace=True)

#
train["y"]=preprocessing.LabelBinarizer().fit_transform(train["y"])
#train.head()

x=train[train.columns.drop('y')]
y= train['y']
x=train[train.columns.drop('y')]
y= train['y']

#
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
     x,y,test_size=0.33, random_state=0)
#----------------------------------------------end of data preprocessing-------------

from sklearn import neighbors as nb

kmax=50

knn_auc_trn1=np.zeros(kmax)
knn_auc_tst1=np.zeros(kmax)
for i in range(3,kmax):
    clf1=nb.KNeighborsClassifier(n_neighbors=i,algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, p=2, weights='uniform' )
    clf1 = clf1.fit(x_train, y_train)
    knn_auc_tst1[i] = mat.roc_auc_score(y_test, clf1.predict_proba(x_test)[:,1])
    knn_auc_trn1[i] = mat.roc_auc_score(y_train, clf1.predict_proba(x_train)[:,1])

pyplot.plot(knn_auc_trn1, linewidth=2, label = "knn train auc")
pyplot.plot(knn_auc_tst1, linewidth=2, label = "knn test auc")
pyplot.legend()
pyplot.xlim(3, 50)
pyplot.ylim(0.8, 1.0)
pyplot.xlabel("Neighbor size")
pyplot.ylabel("AUC Score")
pyplot.show()
