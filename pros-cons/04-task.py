""" Naive Bayes 
Exercise

Tiny spam dataset:

vectorize text
fit MultinomialNB
predict"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

X, y = make_classification(n_samples=200,n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, class_sep=2.0,random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(y_pred)