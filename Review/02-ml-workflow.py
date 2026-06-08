# Machine Learning Workflow

# Load the Titanic dataset
import pandas as pd
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

df = pd.read_csv(url)
print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset information:")
print(df.info())

# Define a Prediction Problem

# Problem: Did the passenger survive?

# Identify Features and Target

y = df["Survived"]
X = df[["Pclass", "Age", "Fare"]]

# Handling Missing Data
print("\nMissing values in Age column:")
print(df["Age"].isnull().sum())
df["Age"] = df["Age"].fillna(
    df["Age"].median()
)
print("\nMissing values in Age column after filling:")
print(df["Age"].isnull().sum())

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose a Machine Learning Model
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    random_state=42
)

model.fit(X_train, y_train)

# Make Predictions and Evaluate the Model
predictions = model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print("\nModel Accuracy:")
# Decision Tree Classifier accuracy on the test set
print(accuracy)

# Compare with a Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("\nRandom Forest Classifier Accuracy:")
print(rf_accuracy)

