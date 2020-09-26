'''
    File name: ML.py
    Author: Kunal Khade
    Date created: 9/20/2020
    Date last modified: 9/24/2020
    Python Version: 3.7

'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Perceptron:

	error = 0
	def __init__(self, learning_rate, iterations):
		#Initialize Instance variables and function
		self.lr = learning_rate
		self.iterations = iterations
		self.active = self.step_input
		self.weights = None
		self.bias = None
		self.error_Val = None

	def fit(self, X, y):
		#Fit method for training data (X) and respected output(y)
		#Training module for perceptron
		n_samples, n_features = X.shape
		self.weights = np.zeros(n_features)
		self.bias = 0
		y_ = np.array([1 if i > 0 else 0 for i in y])
		for _ in range(self.iterations):
			for idx, x_i in enumerate(X):
				lin_out = np.dot(x_i, self.weights)+self.bias
				y_predicted = self.active(lin_out)
				update = self.lr * (y_[idx] - y_predicted)
				self.error_Val =+ abs(lin_out*lin_out)
				self.weights += update * x_i
				self.bias += update
			self.error = np.append(self.error, int(self.error_Val)/n_samples)


	def predict(self, X):
		#Predict the resultant data with respect to training dataset
		lin_out = np.dot(X, self.weights) + self.bias
		y_predicted = self.active(lin_out)
		return y_predicted

	def step_input(self, x):
		#Convert data (x) into -1 and 1
		return np.where(x>=0, 1, -1)

	def net_input(self, X):
		#Display function for net_input
		lin_out = np.dot(X, self.weights) + self.bias
		print(lin_out)

	def plot_decision_regions(self, X, y, classifier, resolution=0.02):
		#Convert complete training data into points
		#plot points on 2d plane
		#Setup marker generator and color map
		markers = ('s', 'x', 'o', '^', 'v')
		colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
		cmap = ListedColormap(colors[:len(np.unique(y))])
		# plot the decision surface
		x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
		x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
		xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
		Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
		Z = Z.reshape(xx1.shape)
		plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
		plt.xlim(xx1.min(), xx1.max())
		plt.ylim(xx2.min(), xx2.max())
		
		# plot class samples
		for idx, cl in enumerate(np.unique(y)):
			plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)
		#Display Result
		plt.xlabel('sepal length')
		plt.ylabel('petal length')
		plt.legend(loc='upper left')
		plt.show()

