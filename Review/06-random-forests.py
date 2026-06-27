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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the Model
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(accuracy)

# Feature Importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

print(feature_importance)

feature_importance.sort_values(
    by="Importance",
    ascending=False
).plot.bar(
    x="Feature",
    y="Importance",
    legend=False
)

plt.title("Feature Importance")
plt.ylabel("Importance")
plt.show()