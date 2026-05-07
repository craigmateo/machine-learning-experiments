""" Decision Trees 
Exercise

Train:

DecisionTreeClassifier
RandomForestClassifier

on same data.

Compare accuracy."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=100,n_features=2,n_redundant=0,n_informative=2,n_clusters_per_class=1,class_sep=2.0,random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X,y)

model = DecisionTreeClassifier(max_depth=2)
model.fit(X_train,y_train)
preds = model.predict(X_test)
print(accuracy_score(y_test, preds))
