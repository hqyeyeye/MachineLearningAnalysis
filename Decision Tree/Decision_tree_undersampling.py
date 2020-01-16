# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 14:23:33 2020

@author: huqio
"""

import pandas as pd
#from imblearn.combine import SMOTETomek,SMOTEENN
from imblearn.under_sampling import NearMiss
import numpy as np
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from sklearn.tree import DecisionTreeClassfier
from sklearn.model_selection import KFold

np.random.seed(0)

df = pd.read_csv('ML_data.csv')

#divide the data set into two sets which has output 0 and the other one has output 1
happen = df[df['CE'] ==1 ]
Nohappen = df[df['CE'] ==0 ]

#select 20% from each data sets to the testing data
happen_train = happen.sample(frac=0.8)
happen_train_drop = happen_train.index
happen_test = happen.drop(happen_train_drop)

Nohappen_train = Nohappen.sample(frac=0.8)
Nohappen_train_drop = Nohappen_train.index
Nohappen_test = Nohappen.drop(Nohappen_train_drop)
 
train_set = pd.concat([happen_train,Nohappen_train])
test_set = pd.concat([happen_test,Nohappen_test])

X_train = train_set.iloc[:, [6,8,9,10,11,13,14,15,16,21]].values
Y_train = train_set.iloc[:, 23].values

X_test = test_set.iloc[:, [6,8,9,10,11,13,14,15,16,21]].values
Y_test = test_set.iloc[:, 23].values

#feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

nm = NearMiss()
X_undsam, Y_undsam = nm.fit_sample(X_train,Y_train) 

clf = DecisionTreeClassfier()
clf = clf.fit(X_undsam,Y_undsam)

Y_pred = clf.predict(X_test)

print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))









