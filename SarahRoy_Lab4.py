# Name: Sarah Ann Roy
# Student ID: 0650615
# COIS 4400H - Lab 4

import pandas as pd
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
df = pd.read_csv('diabetes.csv')
target = [df.columns[-1]]
data= df.columns[0:8]
X = df[data].values
y = df[target].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
ann = MLPClassifier(hidden_layer_sizes=(9,9))
ann.fit(X_train,y_train.ravel())    #or use ann.fit(X_train,y_train.ravel()) if you are getting a conversion warning
predict_test = ann.predict(X_test)
print(confusion_matrix(y_test,predict_test))
print(classification_report(y_test,predict_test))
