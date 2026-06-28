from sklearn.datasets import load_wine

wine = load_wine(as_frame=True)

df = wine.frame

print(df.head())


# Define a Prediction Problem
y = df["target"]

# Identify Features and Target
X = df[["alcohol", "color_intensity"]]

# Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Choose a Machine Learning Model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd 

# Predictions
best_model = None
best_accuracy = 0
best_k = None

for k in [1, 3, 5, 7, 9, 11]:

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"k={k}, accuracy={accuracy:.3f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k
        best_model = model

print("Best k:", best_k)
print("Best accuracy:", best_accuracy)


predictions = best_model.predict(X_test)

cm = confusion_matrix(y_test, predictions)

cm_df = pd.DataFrame(
    cm,
    index=["Actual 0","Actual 1","Actual 2"],
    columns=["Predicted 0","Predicted 1","Predicted 2"]
)

print(cm_df)

# Confusion Matrix

cm = confusion_matrix(y_test, predictions)

cm_df = pd.DataFrame(
    cm,
    index=["Actual 0","Actual 1","Actual 2"],
    columns=["Predicted 0","Predicted 1","Predicted 2"]
)

print(cm_df)

# Visualize the Data

import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))

for target in sorted(df["target"].unique()):

    subset = df[df["target"] == target]

    plt.scatter(
        subset["alcohol"],
        subset["color_intensity"],
        label=f"Class {target}"
    )

plt.xlabel("Alcohol")
plt.ylabel("Color Intensity")
plt.legend()

plt.show()



