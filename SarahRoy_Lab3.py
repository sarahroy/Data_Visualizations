#Name: Sarah Ann Roy
#Student ID: 0650615
#COIS 4400H - Lab 3
#Source: https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

#import libraries
import sklearn.datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, KFold, cross_val_score


X,y = sklearn.datasets.load_wine(return_X_y=True) #loading wine dataset
X_trainingData, X_testData, y_trainingData, y_testData = train_test_split(X,y,test_size=0.33) #dividing and splitting data into train and test sets

#spot checking four algorithms
models = [] #list of classifier models
models.append(('DTC', DecisionTreeClassifier())) #Decision Tree Classifier
models.append(('RFC', RandomForestClassifier())) #Random Forest Classifier
models.append(('KNN', KNeighborsClassifier())) #KNeighbours Classifier
models.append(('GNB', GaussianNB())) #Gaussian Naive Bayes Classifier

#evaluate each classification model
results = [] #list of results
names = [] #list of classifier names
for name, model in models:
    #10 K-Fold Cross Validation
    kfold = KFold(n_splits=10, random_state=None, shuffle=True)
    cv_results = cross_val_score(model, X_trainingData, y_trainingData, cv=kfold, scoring='accuracy')
    results.append(cv_results) #adding the results to the list
    names.append(name) #adding names to the names[] list
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std())) #printing the cross validation scores

#Decision Tree Classifier
dt = DecisionTreeClassifier()
dt.fit(X_trainingData,y_trainingData) #fit data
prediction = dt.predict(X_testData) #predict
#Evaluate predictions of the Decision Tree Classifier
print(confusion_matrix(y_testData, prediction)) #print confusion matrix
print(classification_report(y_testData, prediction)) #print classification report

#Random Forest Classifier
rf = RandomForestClassifier()
rf.fit(X_trainingData, y_trainingData)#fit data
rf_predict = rf.predict(X_testData)#predict
#Evaluate predictions of the Random Forest Classifier
print(confusion_matrix(y_testData, rf_predict))#print confusion matrix
print(classification_report(y_testData, rf_predict))#print classification report


#KNeighboursClassifier
knn = KNeighborsClassifier()
knn.fit(X_trainingData, y_trainingData)#fit data
knn_predict = knn.predict(X_testData)#predict
#Evaluate predictions of the KNeighboursClassifier
print(confusion_matrix(y_testData, knn_predict))#print confusion matrix
print(classification_report(y_testData, knn_predict))#print classification report


#Gaussian Naive Bayes Classifier
gnb = GaussianNB()
gnb.fit(X_trainingData, y_trainingData)#fit data
gnb_predict = gnb.predict(X_testData)#predict
#Evaluate predictions of the Gaussian Naive Bayes Classifier
print(confusion_matrix(y_testData, gnb_predict))#print confusion matrix
print(classification_report(y_testData, gnb_predict))#print classification report
