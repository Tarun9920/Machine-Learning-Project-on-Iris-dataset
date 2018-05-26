# -*- coding: utf-8 -*-
"""
Created on Sat May 26 16:24:19 2018

@author: tarun
"""

import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as kn
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score as ac
#plt.style.use('ggplot')
iris= datasets.load_iris()
type(iris) 
print(iris.keys())
iris.data.shape
iris.target_names
 # EDA
X= iris.data
y= iris.target
y= y.astype(float)
df= pd.DataFrame(X,columns= iris.feature_names)
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size= 1/3,random_state= 6)

knn= kn(n_neighbors= 6)
knn.fit(X_train,y_train)
pred= knn.predict(X_test)
#visualization

#plt.scatter(X_train,y_train,color='red')
#plt.plot(X_train,knn.predict(X_train),color= 'blue')
print("Accuracy using Knn AT N=6 is: ")
ac(y_test,pred)

#print(df.head())
#pd.scatter_matrix(df,c=y,figsize= [9,9],s= 150,marker= 'D')