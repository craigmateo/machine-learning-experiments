import pandas as pd
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)

# Load the California housing dataset
df = housing.frame

# Explore the dataset
print(df.head())
print(df.info())

# Define a Prediction Problem
X = df[["MedInc", "AveRooms", "HouseAge"]]
y = df["MedHouseVal"]

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Choose a Machine Learning Model
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

# Make Predictions and Evaluate the Model
predictions = model.predict(X_test)

# Print the first 10 predictions and the corresponding true values
print("Predictions:", predictions[:10])
print("True Values:", y_test.head(10))

# Create a DataFrame to compare actual vs predicted values
print("\nComparison of Actual vs Predicted values:")
comparison = pd.DataFrame({
    "Actual": y_test.values[:10],
    "Predicted": predictions[:10]
})

print(comparison)

# Evaluate the model using MAE and R²
from sklearn.metrics import mean_absolute_error, r2_score

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("MAE:", mae)
print("R²:", r2)

# Visualize the results
import matplotlib.pyplot as plt

plt.scatter(y_test, predictions)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--"
)

plt.xlabel("Actual")
plt.ylabel("Predicted")

plt.show()