import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import time

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

# Naive Bayes on Clean, Independent Features

""" In this first experiment, we’ll generate a synthetic dataset with two informative features that are conditionally independent given the class label. 
This aligns well with the assumptions made by Naive Bayes, so we expect the model to perform well.

We’ll fit a Gaussian Naive Bayes model to the training data and then visualize its decision boundary and confusion matrix on the test set.

Pay attention to how neatly it separates the two classes and how confident its predictions are. """

# Generate linearly separable data
X, y = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0,
                           n_clusters_per_class=1, class_sep=2.0, random_state=0)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train the Naive Bayes classifier
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Plot decision boundary with true labels
fig, ax = plt.subplots(figsize=(6, 4))
plot_decision_boundary(model, X_test, y_pred, ax, "Naive Bayes on Clean/Separable Data")
plt.show()

# Show accuracy and confusion matrix
acc = accuracy_score(y_test, y_pred)
disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.title(f"Confusion Matrix - Clean/Separable Data\nAccuracy: {acc:.2f}")
plt.show()

#Naive Bayes on Correlated or Noisy Features¶

""" Now let’s evaluate Naive Bayes on a more challenging dataset.

Here, the features are not independent—one is a redundant version of the other, and the data has overlapping class boundaries.

This violates the core assumption of Naive Bayes and may cause the model to misestimate class probabilities. 
We’ll train the model again and inspect how the predictions and decision boundaries are affected.

Expect to see lower accuracy and a more ambiguous classification boundary. """

# Generate data with correlated features and some noise
X_corr, y_corr = make_classification(n_samples=200, n_features=2, n_informative=1, n_redundant=1,
                                     n_clusters_per_class=1, class_sep=0.8, random_state=1)

# Split the data
X_train_corr, X_test_corr, y_train_corr, y_test_corr = train_test_split(X_corr, y_corr, test_size=0.25, random_state=42)

# Train Gaussian Naive Bayes model
model_corr = GaussianNB()
model_corr.fit(X_train_corr, y_train_corr)
y_pred_corr = model_corr.predict(X_test_corr)

# Plot decision boundary
fig, ax = plt.subplots(figsize=(6, 4))
plot_decision_boundary(model_corr, X_test_corr, y_pred_corr, ax, "Naive Bayes on Correlated/Noisy Data")
plt.show()

# Show accuracy and confusion matrix
acc_corr = accuracy_score(y_test_corr, y_pred_corr)
disp_corr = ConfusionMatrixDisplay.from_estimator(model_corr, X_test_corr, y_test_corr)
plt.title(f"Confusion Matrix - Correlated/Noisy Data\nAccuracy: {acc_corr:.2f}")
plt.show()

