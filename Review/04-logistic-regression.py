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
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Make Predictions and Evaluate the Model
predictions = model.predict(X_test)

# Evaluate the Model
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)

print(accuracy)

# Get the predicted probabilities for each class
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions,
    "Prob_Died": probabilities[:, 0],
    "Prob_Survived": probabilities[:, 1]
})

print(results.head(20))

# Visualize the predicted probabilities
import matplotlib.pyplot as plt

plt.hist(probabilities[:,1], bins=20)

plt.xlabel("Probability of Survival")
plt.ylabel("Passengers")
plt.title("Predicted Survival Probabilities")

plt.show()

# Interpret the Model
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

print(coef_df)

coef_df.plot.bar(
    x="Feature",
    y="Coefficient",
    legend=False
)