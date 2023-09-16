#Name: Sarah Ann Roy
#Student Number: 0650615
#COIS 4400H: Lab 5
#Source: https://fangya18.com/2018/12/19/clustering-analysis-iris-dataset/

from sklearn.datasets import load_iris
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans,DBSCAN
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram

################################Part 1: Determining the number of clusters for Iris data ###############################
data = load_iris() #loading the iris dataset from sklearn
numData = data['data']
scaledData = preprocessing.scale(numData)

SSE=[] #list of sum of the squared error

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, max_iter=300)
    kmeans.fit(scaledData)
    SSE.append(kmeans.inertia_)

plt.title('COIS 4400 Lab 5 - Sarah Ann Roy (0650615) \n SSE vs Number of Clusters') #title of the plot
plt.xlabel('Number of Clusters')  #label x axis
plt.ylabel('SSE') #label y axis
plt.plot(range(1,11),SSE)
plt.show() #display plot

########################################## Part 2: Additional Clustering methods #######################################
# Method #1: Hierachy Clustering
x=data.data
y=data.target
hier=linkage(x,"ward")
max_d=7.08
plt.figure(figsize=(25,10))
plt.title('COIS 4400 Lab 5 - Sarah Ann Roy (0650615) \n Iris Hierarchical Clustering Dendrogram')#title of the plot
plt.xlabel('Species')#label x axis
plt.ylabel('distance')#label y axis
dendrogram(
    hier,
    truncate_mode='lastp',
    p=50,
    leaf_rotation=90.,
    leaf_font_size=8.,
)
plt.axhline(y=max_d, c='k')
plt.show()#display plot

# Method #2: DBSCAN
dbscan=DBSCAN()
dbscan.fit(x)
pca=PCA(n_components=2).fit(x)
pca_2d=pca.transform(x)
for i in range(0, pca_2d.shape[0]):
    if dbscan.labels_[i] == 0:
        c1 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', marker='+')
    elif dbscan.labels_[i] == 1:
        c2 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='g', marker='o')
    elif dbscan.labels_[i] == -1:
        c3 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='b', marker='*')
plt.legend([c1, c2, c3], ['Cluster 1', 'Cluster 2', 'Noise']) #legends
plt.title('COIS 4400 Lab 5 - Sarah Ann Roy (0650615) \n DBSCAN finds 2 clusters and Noise') #title of the plot
plt.show()#display plot