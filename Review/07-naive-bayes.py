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
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

model = GaussianNB()

# Train the Model
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(accuracy)

# Feature Importance
probabilities = model.predict_proba(X_test)
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions,
    "Prob_Died": probabilities[:,0],
    "Prob_Survived": probabilities[:,1]
})

print(results.head())

# Confusion Matrix
from sklearn.metrics import confusion_matrix
import pandas as pd

cm = confusion_matrix(y_test, predictions)

cm_df = pd.DataFrame(
    cm,
    index=["Actual Died", "Actual Survived"],
    columns=["Predicted Died", "Predicted Survived"]
)

print(cm_df)

# Visualize the Confusion Matrix
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Died", "Survived"],
    yticklabels=["Died", "Survived"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()