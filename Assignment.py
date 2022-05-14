# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 09:48:51 2022

@author: Balaji
"""

import numpy as np
import pandas as pd

from sklearn import preprocessing , datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, Normalizer , LabelEncoder

x=np.random.random((10,5))  #dataset creation
print(x)

y=np.array(['M','F','N','F','M','N','F','N','M','N'])

print(x.shape)
print(y.shape)

x_train, x_test , y_train , y_test = train_test_split(x, y, test_size=0.3, random_state=0)

scaler = StandardScaler()
                                                  #preprocessing
a=np.array([1,2,3,4,5,30,50])                     #1)Standardization
 
print("Mean:",a.mean())
print("SD: ",a.std())

#a_std=scaler.fit_transform(a.reshape(-1,1))
#print("Mean:",a.mean())
#print("SD: ",a.std())

xtrain_std=scaler.fit_transform(x_train)
xtest_std=scaler.transform(x_test)

print(xtrain_std.mean(), xtest_std.std())

from sklearn.preprocessing import LabelEncoder
enco=LabelEncoder()

y_enc=enco.fit_transform(y_train)
yt_enc = enco.fit_transform(y_test)

#supervised

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor

#unsupervised

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

#MOdel fitting

mlp = MLPClassifier()
print(xtrain_std.shape , y_enc.shape)

mlp.fit(xtrain_std,y_enc)


print(mlp.score(xtrain_std, y_enc))
print(mlp.score(xtest_std,yt_enc ))

y_pred = mlp.predict(x_test)
print(y_pred)


from sklearn.metrics import classification_report

cm = confusion_matrix(yt_enc, y_pred)
print(cm)


cr = classification_report(yt_enc, y_pred)
print(cr)















