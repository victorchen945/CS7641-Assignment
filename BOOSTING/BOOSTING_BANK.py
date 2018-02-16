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

from sklearn import ensemble as bst
from sklearn import tree
from sklearn import neural_network as nw
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
    

pyplot.title('Ensembled vs weak learner_ bank dataset')
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



