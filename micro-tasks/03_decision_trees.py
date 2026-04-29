import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# For reproducibility
np.random.seed(0)

def plot_decision_boundary(model, X, y, ax, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, cmap=ListedColormap(['#FFBBBB', '#BBBBFF']), alpha=0.4)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k', alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

## Decision Tree on Simple Data

""" Let’s start by training a decision tree on a synthetic dataset that is clean and linearly separable. 
This type of data allows us to clearly see how the tree splits the feature space based on informative rules.

In this scenario, we expect the decision tree to perform well, both in terms of accuracy and interpretability. 
You’ll visualize the 2D decision boundary, then inspect the corresponding tree structure to see how each decision is made at various levels of the tree. """

# Generate data
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                           n_informative=2, n_clusters_per_class=1,
                           class_sep=2.0, random_state=0)

# Train Decision Tree
model = DecisionTreeClassifier(max_depth=3)
model.fit(X, y)

# Plot decision boundary
fig, ax = plt.subplots(figsize=(8, 6))
plot_decision_boundary(model, X, y, ax, "Decision Tree on Linearly Separable Data")
plt.show()

# Plot tree structure
plt.figure(figsize=(10, 6))
plot_tree(model, filled=True, feature_names=["Feature 1", "Feature 2"])
plt.title("Tree Structure")
plt.show()

## Decision Tree on Noisy or Complex Data

""" Now let’s examine how a decision tree behaves with data that is more difficult to separate. 
This dataset includes noise and overlapping classes, which makes it challenging to learn a perfect boundary.

In such settings, a decision tree may grow deep and overfit the training data — memorizing specific patterns rather than generalizing well. 
We’ll train a deep, unrestricted tree and observe how complex the resulting structure becomes. """

# Generate complex dataset
X_complex, y_complex = make_classification(n_samples=100, n_features=2, n_redundant=0,
                                           n_informative=2, n_clusters_per_class=1,
                                           class_sep=0.5, flip_y=0.3, random_state=1)

# Train a decision tree without max_depth limitation
model_complex = DecisionTreeClassifier(max_depth=None)
model_complex.fit(X_complex, y_complex)

# Plot decision boundary
fig, ax = plt.subplots(figsize=(8, 6))
plot_decision_boundary(model_complex, X_complex, y_complex, ax, "Decision Tree on Complex Data")
plt.show()

# Visualize tree structure
plt.figure(figsize=(12, 6))
plot_tree(model_complex, filled=True, feature_names=["Feature 1", "Feature 2"])
plt.title("Deep Tree Structure (Potential Overfitting)")
plt.show()