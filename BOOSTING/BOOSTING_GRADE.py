# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 01:31:09 2018

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

#--------------------------------------END OF PREPROCESSING----------------------

from sklearn import ensemble as bst
from sklearn import tree
from matplotlib import pyplot


base_tr=tree.DecisionTreeClassifier(max_depth=3,max_leaf_nodes=2)
base_tr.fit(x_train,y_train)
trscn=mat.roc_auc_score(y_test, base_tr.predict_proba(x_test)[:,1])
trsct= (mat.roc_auc_score(y_train, base_tr.predict_proba(x_train)[:,1]))

max_learner=100

ada_auc_tst = np.zeros(max_learner)
ada_auc_trn = np.zeros(max_learner)
tree_auc_tst= np.zeros(max_learner)
tree_auc_trn= np.zeros(max_learner)
for i in range(1,max_learner):
    ada = bst.AdaBoostClassifier(base_estimator=base_tr,learning_rate=1,
                                 n_estimators=i,algorithm="SAMME.R")
    ada.fit(x_train, y_train)
    ada_auc_tst[i] = mat.roc_auc_score(y_test, ada.predict_proba(x_test)[:,1])
    ada_auc_trn[i] = mat.roc_auc_score(y_train, ada.predict_proba(x_train)[:,1])
    tree_auc_tst[i]= trsct
    tree_auc_trn[i]= trscn
    

pyplot.title('Ensembled vs weak learner _ Grade dataset')
pyplot.plot(ada_auc_trn, linewidth=2, label = "adaboost train")
pyplot.plot(ada_auc_tst, linewidth=2, label = "adaboost test")
pyplot.plot(tree_auc_trn, linewidth=2, label = "weaklearner train")
pyplot.plot(tree_auc_tst, linewidth=2, label = "weaklearner test")
pyplot.legend()
pyplot.xlim(0,100)
pyplot.ylim(0.6, 1.0)
pyplot.xlabel("n_estimators")
pyplot.ylabel("AUC Scores")
pyplot.show()

