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
from sklearn.svm import SVC

svm_model = SVC(
    kernel="linear",        
    probability=True,
    random_state=42 
)

# Train the Model
svm_model.fit(X_train, y_train)

# Make Predictions
predictions = svm_model.predict(X_test)

# Evaluate the Model
from sklearn.metrics import accuracy_score
import pandas as pd

accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy:.3f}")

# Feature Importance
probabilities = svm_model.predict_proba(X_test)

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions,
    "Prob_Class_0": probabilities[:,0],
    "Prob_Class_1": probabilities[:,1],
    "Prob_Class_2": probabilities[:,2]
})

print(results.head(10))

# Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predictions)

cm_df = pd.DataFrame(
    cm,
    index=["Actual 0","Actual 1","Actual 2"],
    columns=["Predicted 0","Predicted 1","Predicted 2"]
)

print(cm_df)

# Scatter Plot of Features
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