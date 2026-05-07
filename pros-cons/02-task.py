""" Perceptron / Classification
Exercise
create synthetic classification data
fit Perceptron
predict classes"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=100,n_features=2,n_redundant=0,n_informative=2, n_clusters_per_class=2,class_sep=2.0,random_state=0)

y = 2 * y -1

model = Perceptron(max_iter=1000)
model.fit(X,y)

print(model.predict([[2, 3]]))

plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr', edgecolors='k')
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Create decision boundary
xx = np.linspace(xlim[0], xlim[1])
w = model.coef_[0]
b = model.intercept_[0]
yy = -(w[0] * xx + b) / w[1]

plt.plot(xx, yy, 'k--')
plt.title("Perceptron on Linearly Separable Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

