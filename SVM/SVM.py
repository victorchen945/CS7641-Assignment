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

x_scaled=preprocessing.scale(x)

# divide dataset into train set and test set, size of test equals = 0.33*size of dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
     x_scaled,y,test_size=0.33, random_state=0)

#--------------------------------------END OF PREPROCESSING----------------------

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
    svmclf.fit(x_train,y_train)
    svm_auc_tst[i] = mat.roc_auc_score(y_test, svmclf.predict_proba(x_test)[:,1])
    svm_auc_trn[i] = mat.roc_auc_score(y_train, svmclf.predict_proba(x_train)[:,1])
pyplot.title('Scores by costs')
pyplot.plot(svm_auc_trn, linewidth=2, label = "svm train auc")
pyplot.plot(svm_auc_tst, linewidth=2, label = "svm test auc")
pyplot.xticks(range(len(costs)),['0.00001','0.0001','0.001','0.01','0.1','0','10','100','1000','10000','100000'])
pyplot.legend()
pyplot.ylim(0.8, 1.0)
pyplot.xlabel("Costs")
pyplot.ylabel("AUC Scores")
pyplot.show()

