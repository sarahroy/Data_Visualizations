#Name: Sarah Ann Roy
#Student Number: 0650615
#COIS 4400H Lab 1
#Sources: https://benalexkeen.com/scatter-charts-in-matplotlib/, https://towardsdatascience.com/matplotlib-tutorial-with-code-for-pythons-powerful-data-visualization-tool-8ec458423c5e

#import packages
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets

#load iris data
iris = datasets.load_iris()
iris_df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
iris_df['species'] = iris['target']

colours = ['indigo', 'purple', 'magenta', 'violet'] #list of colours
species = ['I. setosa', 'I. versicolor', 'I. virginica'] #list of species

#Scatter Plot : Sepal Length vs Petal Length
for i in range(0, 3):
    species_df = iris_df[iris_df['species'] == i]
    plt.scatter( #scatter plot
        species_df['sepal length (cm)'], #x-coordinates
        species_df['petal length (cm)'], #y-coordinates
        color=colours[i], #pick a colour from the list of colours
        alpha=0.5,
        label=species[i] #pick labels from the list of species
    )

plt.xlabel('sepal length (cm)') #label x axis
plt.ylabel('petal length (cm)') #label y axis
plt.title('COIS 4400 Lab 1 - Sarah Ann Roy (0650615) \n Scatter Plot: Petal Length vs Sepal Length') #scatter plot title
plt.legend(loc='lower right')
plt.show() #display scatter plot

#Bar Plot : Sentosa Averages
X_iris = iris.data #x axis
Y_iris = iris.target #y axis
average = X_iris[Y_iris == 0].mean(axis=0) #average of values

plt.bar(iris.feature_names, average, color = colours) #bar plot
plt.title('COIS 4400 Lab 1 - Sarah Ann Roy (0650615) \n Bar Plot: Setosa Averages') #bar plot title
plt.ylabel('Average (cm)') #label y axis
plt.show() #display bar plot