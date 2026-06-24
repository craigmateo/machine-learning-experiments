# Load the Titanic dataset
import pandas as pd
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

df = pd.read_csv(url)
print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset information:")
print(df.info())

# Define a Prediction Problem
y = df["Survived"]

# Identify Features and Target
X = df[["Pclass", "Age", "Fare"]]


# Handling Missing Data
X["Age"] = X["Age"].fillna(
    X["Age"].median()
)

# Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Choose a Machine Learning Model
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

best_accuracy = 0
best_depth = None
best_model = None

#for depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None]:
for depth in [1, 2, 3, 4, 5, None]:

    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Depth={depth}, Accuracy={accuracy:.3f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_depth = depth
        best_model = model

# Evaluate the Model
model = best_model

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

print("Best depth:", best_depth)
print("Best accuracy:", best_accuracy)
print("Actual depth:", model.get_depth())
print("Leaves:", model.get_n_leaves())

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions,
    "Prob_Died": probabilities[:, 0],
    "Prob_Survived": probabilities[:, 1]
})

print(results.head(20))

# Visualize the predicted probabilities
plt.figure(figsize=(15, 10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Died", "Survived"],
    filled=True
)

plt.show()

# Interpret the Model
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

print(feature_importance)

feature_importance.plot.bar(
    x="Feature",
    y="Importance",
    legend=False
)

plt.show()

# Evaluate the Model with a Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predictions)

print(cm)