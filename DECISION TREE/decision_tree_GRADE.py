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
for v in var:
    print('\nFrequency count for variable %s'%v) 
    print(dataset[v].value_counts())
    
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

#-----------------------------------END OF PREPROCESSING

learner=tree.DecisionTreeClassifier("gini",max_depth=5)
learner.fit(x_train, y_train)

result=learner.predict(x_test)
#print (result)

from IPython.display import Image

def plot_tree(clf,feature_names):
    tree.export_graphviz(learner,out_file='tree.dot',class_names=['No','Yes'],feature_names=feature_names, 
                         filled=True, rounded=True, special_characters=True, proportion=True)
    os.system("dot -Tpng tree.dot -o tree.png")
    os.system("tree.png")
plot_tree(learner, x_train.columns)
# Note : Uncoverted Quotes (Yes) and Converted quotes (No)
Image(filename='tree.png')


maxdepth = 50
tree_auc_trn= np.zeros(maxdepth)
tree_auc_tst= np.zeros(maxdepth)
for i in range(1,maxdepth):
    clf1 = tree.DecisionTreeClassifier(criterion="gini",max_depth=i)
    clf1 = clf1.fit(x_train, y_train)
    tree_auc_trn[i] = mat.roc_auc_score(y_test, clf1.predict_proba(x_test)[:,1])
    tree_auc_tst[i] = mat.roc_auc_score(y_train, clf1.predict_proba(x_train)[:,1])

from matplotlib import pyplot
pyplot.plot(tree_auc_tst, linewidth=3, label = "Decision tree test AUC")
pyplot.plot(tree_auc_trn, linewidth=3, label = "Decision tree train AUC")
pyplot.legend()
pyplot.ylim(0.5, 1.0)
pyplot.xlabel("Max_depth")
pyplot.ylabel("validation auc")
#plt.figure(figsize=(12,12))
pyplot.show()


para = 30
tree_auc_trn = np.zeros(para)
tree_auc_tst = np.zeros(para)
for i in range(2,para):
    #tree.DecisionTreeClassifier(criterion=,min_samples_leaf=)
    clf1 = tree.DecisionTreeClassifier(criterion="gini",max_leaf_nodes=i)
    clf1 = clf1.fit(x_train, y_train)
    tree_auc_trn[i] = mat.roc_auc_score(y_test, clf1.predict_proba(x_test)[:,1])
    tree_auc_tst[i] = mat.roc_auc_score(y_train, clf1.predict_proba(x_train)[:,1])

from matplotlib import pyplot
pyplot.plot(tree_auc_tst, linewidth=3, label = "Decision tree test AUC")
pyplot.plot(tree_auc_trn, linewidth=3, label = "Decision tree train AUC")
pyplot.legend()
pyplot.ylim(0.8, 1.0)
pyplot.xlabel("Max_depth")
pyplot.ylabel("validation auc")
#plt.figure(figsize=(12,12))
pyplot.show()

