# Name: Sarah Ann Roy
# Student ID: 0650615
# COIS 4400H - Lab 6
# Sources: https://fangya18.com/2018/12/19/clustering-analysis-iris-dataset/
# https://gist.github.com/spyhi/ec8e60419d90aefc8537eb557ef35826
# https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py

from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import load_breast_cancer, load_wine, load_iris
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.datasets import make_blobs

################################ Part 1 - Supervised Method: Decision Tree Classifier ##################################
X,y = load_breast_cancer(return_X_y=True) #load breast cancer dataset
X_trainingData, X_testData, y_trainingData, y_testData = train_test_split(X,y,test_size=0.33) #dividing and splitting data into train and test sets

# Method 1: Decision Tree Classifier
dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dt.fit(X_trainingData,y_trainingData) #fit data
prediction = dt.predict(X_testData) #predict
# Evaluate predictions of the Decision Tree Classifier
print(confusion_matrix(y_testData, prediction)) #print confusion matrix
print(classification_report(y_testData, prediction)) #print classification report

# Method 2: Random Forest Classifier
rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rf.fit(X_trainingData, y_trainingData)#fit data
rf_predict = rf.predict(X_testData)#predict
# Evaluate predictions of the Random Forest Classifier
print(confusion_matrix(y_testData, rf_predict))#print confusion matrix
print(classification_report(y_testData, rf_predict))#print classification report


############################################# Part 2 - Unsupervised Methods ############################################
# Method 1: KMeans Clustering
data = pd.read_csv('breast_cancer_dataset.csv') #load dataset as a dataframe
feat_cols = ["radius_mean", "texture_mean", "texture_mean", "area_mean", \
                      "smoothness_mean", "compactness_mean", "concavity_mean", \
                      "symmetry_mean", "fractal_dimension_mean"]
feat_cols_sm = ["radius_mean", "concavity_mean", "symmetry_mean"]
features = np.array(data[feat_cols_sm])
clusters = KMeans(n_clusters=2, max_iter=300) # Initialize the KMeans cluster module
clusters.fit(features)
centroids = clusters.cluster_centers_
labels = clusters.labels_
print(centroids)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = ["r", "b"] #red and blue
# Plot all the features and assign color based on cluster identity label
for i in range(len(features)):
    ax.scatter(xs=features[i][0], ys=features[i][1], zs=features[i][2], c=colors[labels[i]], zdir='z')
ax.scatter(xs=centroids[:,0], ys=centroids[:,1], zs=centroids[:,2], marker="x", s=150, c="c") # Plot centroids
diag = np.array(data['diagnosis']) # Create array of diagnosis data, which should be same length as labels.
matches = 0# Create variable to hold matches in order to get percentage accuracy.
# Transform diagnosis vector from B||M to 0||1 and matches++ if correct.
for i in range(0, len(diag)):
    if diag[i] == "B":
        diag[i] = 0
    if diag[i] == "M":
        diag[i] = 1
    if diag[i] == labels[i]:
        matches = matches + 1

#Calculate percentage matches and print.
percentMatch = (matches/len(diag))*100
print("Percent matched between benign and malignant ", percentMatch)

#Set labels on figure and show 3D scatter plot to visualize data and clusters.
ax.set_xlabel("Radius Mean")
ax.set_ylabel("Concavity Mean")
ax.set_zlabel("Symmetry Mean")
plt.title('COIS 4400 Lab 6 - Sarah Ann Roy (0650615) \n KMeans Clustering on Breast Cancer Dataset') #title of the plot
plt.show()


# Method 2: DBSCAN
# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(
    n_samples=750, centers=centers, cluster_std=0.4, random_state=0
)
X = StandardScaler().fit_transform(X)
# Compute DBSCAN
db = DBSCAN(eps=0.3, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
print(
    "Adjusted Mutual Information: %0.3f"
    % metrics.adjusted_mutual_info_score(labels_true, labels)
)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=14,
    )

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )

plt.title("COIS 4400 Lab 6 - Sarah Ann Roy (0650615) \n DBSCAN on make_blobs dataset \n Estimated number of clusters: %d" % n_clusters_)
plt.show()