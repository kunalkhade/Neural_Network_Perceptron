'''
    File name: test.py
    Supporting file: ML.py
    Author: Kunal Khade
    Date created: 9/20/2020
    Date last modified: 9/24/2020
    Python Version: 3.7

    Topic: Develop generic binary classifier perceptron 
    class in ML.py.  It has to taketraining  set  of  any  size.   
    Class  must  include  four  functions  :init(),  fit()  ,netinput(), 
    predict(), One more supportive function to display result.

'''
from ML import Perceptron
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.colors import ListedColormap

#pn instance variable pn assign to Perceptron class
#Pass learning rate = 0.25 and Iteration = 10
pn = Perceptron(0.1, 10)

#Using Pandas import Iris dataset
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

#Only use initial 100 data value labels 
y = df.iloc[0:100, 4].values

#Convert labels into -1 and 1
y = np.where(y == 'Iris-setosa', -1, 1)

#Extract only 2 parameters from data set
X = df.iloc[0:100, [0, 2]].values

#Use fit, error, predict, weights, net_input functions from perceptron class
pn.fit(X, y)
print("Errors : \n", pn.error)
print("Prediction : \n",pn.predict(X)) 
print("Weights : \n", pn.weights)
#print(pn.net_input(X))

#Plot result 
pn.plot_decision_regions(X, y, classifier=pn, resolution=0.02)

#Plot Error function 
plt.plot(range(1, len(pn.error) + 1), pn.error,
marker='o')
plt.xlabel('Iteration')
plt.ylabel('# of misclassifications')
plt.show()