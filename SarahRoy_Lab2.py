#Name: Sarah Ann Roy
#Student ID: 0650615
#COIS 4400H - Lab 2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle

#Part 2: Preprocessing and Determining the number of clusters

california = sklearn.datasets.fetch_california_housing() #load california dataset
boston = sklearn.datasets.load_boston() #load boston dataset
df = pd.DataFrame(boston.data, columns = boston.feature_names)
df["MEDV"] = boston.target
X = df.drop("MEDV",1)   #Feature Matrix
y = df["MEDV"]          #Target Variable
df.head()
numData = boston['data']
scaledData = preprocessing.scale(numData)

SSE=[]# list to keep track of how good the results are

for i in range(1,9):
    kmeans = KMeans(n_clusters=i, max_iter=300)
    kmeans.fit(scaledData)
    SSE.append(kmeans.inertia_)

plt.plot(range(1,9),SSE)
plt.title('COIS 4400 Lab 2 - Sarah Ann Roy (0650615) \n Plot: Number of Clusters vs SSE')
plt.ylabel('SSE') #label y axis
plt.xlabel('Number of Clusters') #label x axis
plt.show()

kmeans = KMeans(n_clusters=3, max_iter=300)
kmeans.fit(scaledData)
predictions = kmeans.predict(scaledData)

plt.scatter(scaledData[:,0],scaledData[:,1], c = predictions)
plt.title('COIS 4400 Lab 2 - Sarah Ann Roy (0650615) \n Scatter Plot')
plt.show()

#Part 3: Decision Tree
X,y = sklearn.datasets.load_wine(return_X_y=True) #I kept getting errors using Boston Housing prices and California Housing prices, this is the only dataset that printed properly other than iris
X= shuffle(X)
X_trainingData, X_testData, y_trainingData, y_testData = train_test_split(X,y,test_size=0.33)

myClassifier=DecisionTreeClassifier()
myClassifier.fit(X_trainingData,y_trainingData)
prediction=myClassifier.predict(X_testData)

print(classification_report(y_testData,prediction))
print(confusion_matrix(y_testData,prediction))
