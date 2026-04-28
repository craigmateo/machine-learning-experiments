import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# For reproducibility
np.random.seed(0)

# Linearly Separable Data

# Apply the Perceptron algorithm to a synthetic dataset where the two classes can be perfectly separated by a straight line. 
# In this scenario, the Perceptron is guaranteed to find a decision boundary that correctly classifies all training points after a finite number of updates.

X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2,
                           n_clusters_per_class=1, class_sep=2.0, random_state=0)

y = 2 * y - 1  # convert labels from {0,1} to {-1,1}

model = Perceptron(max_iter=1000)
model.fit(X, y)

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

# Non-Linearly Separable Data

# Now let’s try the Perceptron on data that is not linearly separable. 
# The Perceptron will attempt to find a linear boundary, but it won’t be able to classify all points correctly.