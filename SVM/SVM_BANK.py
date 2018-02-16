# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 23:48:47 2018

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
x=x[0:300]
y=y[0:300]

x_scaled=preprocessing.scale(x)
#
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
     x_scaled,y,test_size=0.33, random_state=0)
#----------------------------------------------end of data preprocessing-------------


from matplotlib import pyplot
from sklearn import svm

svm_auc_trn,svm_auc_tst,act_clfscore=[],[],[]
kernels=['linear','poly','rbf','sigmoid']
for k in kernels:
    svmclf=svm.SVC(kernel=k,probability=True)
    svmclf = svmclf.fit(x_train, y_train)
    act_clfscore.append(mat.roc_auc_score(y_test, svmclf.predict_proba(x_test)[:,1]))

p1 = pyplot.bar([1,2,3,4], act_clfscore, 0.4, color='#d62728')
pyplot.ylabel('Scores')
pyplot.title('Scores by different kernels')
pyplot.xticks([1,2,3,4], kernels)
pyplot.yticks(np.arange(0, 1.01, 0.02))
pyplot.ylim(0.9, 1.0)
pyplot.show()

costs = np.power(10.0, range(-5,5))
svm_auc_tst = np.zeros(len(costs))
svm_auc_trn = np.zeros(len(costs))

for i in range(len(costs)):
    svmclf = svm.SVC(kernel = 'linear', C=costs[i], probability=True)
    svmclf=svmclf.fit(x_train,y_train)
    svm_auc_tst[i] = mat.roc_auc_score(y_test, svmclf.predict_proba(x_test)[:,1])
    svm_auc_trn[i] = mat.roc_auc_score(y_train, svmclf.predict_proba(x_train)[:,1])
pyplot.title('Scores by costs_ bank dataset')
pyplot.plot(svm_auc_trn, linewidth=2, label = "svm train auc")
pyplot.plot(svm_auc_tst, linewidth=2, label = "svm test auc")
pyplot.xticks(range(len(costs)),['0.00001','0.0001','0.001','0.01','0.1','0','10','100','1000','10000','100000'])
pyplot.legend()
pyplot.ylim(0.8, 1.0)
pyplot.xlabel("Costs")
pyplot.ylabel("AUC Scores")
pyplot.show()

