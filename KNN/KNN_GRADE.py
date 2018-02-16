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



#--------------------------------------END OF PREPROCESSING----------------------
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











