""" 

==========Code==============

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("data.csv")

X = df[["feature_1", "feature_2", "feature_3"]]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(accuracy) 

==========Objects=========

df = whole table
X = input features
y = target
X_train = features used for learning
X_test = features held back
model = algorithm object
predictions = model’s guesses
accuracy = comparison between guesses and true answers

"""
# Example of loading a dataset and inspecting it
## load Iris dataset
import pandas as pd

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset information:")
print(df.info())

# Example of filtering data based on a condition

sl = df[["sepal_length"]]
above_7 = sl[sl["sepal_length"] > 7]
print("\nSepal lengths greater than 7:")
print(above_7.head())

# Check for missing values in the dataset
print("\nMissing values:")
print(df.isnull().sum())

print("\nDataset description:")
df_clean = df.dropna()
print("\nDataset after dropping missing values:")
print(df_clean.head())

# Fill missing values in 'sepal_width' with the mean of that column
df["sepal_width"] = df["sepal_width"].fillna(
    df["sepal_width"].mean()
)

# Create a new feature 'total_size' by summing the lengths and widths of sepals and petals
df["total_size"] = (
    df["sepal_length"] +
    df["sepal_width"] +
    df["petal_length"] +
    df["petal_width"]
)
print("\nDataset with new feature 'total_size':")
print(df.head())

# Create a new feature 'petal_ratio' by dividing petal length by petal width
df["petal_ratio"] = df["petal_length"] / df["petal_width"]
print("\nDataset with new feature 'petal_ratio':")
print(df.head())

# Group the data by species and calculate the mean sepal length for each species
import matplotlib.pyplot as plt
df.groupby("species")["sepal_length"].mean()
print("\nMean sepal length by species:")
print(df.groupby("species")["sepal_length"].mean())

# Export the cleaned dataset to a new CSV file
df.to_csv("cleaned_iris.csv", index=False)

# Visualize the distribution of sepal length using a histogram
import matplotlib.pyplot as plt
df[["sepal_length"]].hist()
plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Frequency")
plt.show()

# Create a histogram of sepal length with 20 bins
df["sepal_length"].hist(bins=20)
plt.show()

# Create overlapping histograms of sepal length for each species
for species in df["species"].unique():
    df[df["species"] == species]["sepal_length"].hist(
        alpha=0.5,
        bins=10,
        label=species
    )

plt.legend()
plt.show()

# Create a scatter plot of sepal length vs sepal width, colored by species
df.plot.scatter(x="sepal_length", y="sepal_width", c="species", cmap="viridis") 
plt.title("Sepal Length vs Sepal Width")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()

# Create a scatter plot of petal length vs petal width, colored by species
df.plot.scatter(x="petal_length", y="petal_width", c="species", cmap="viridis") 
plt.title("Petal Length vs Petal Width")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()

# Create a box plot to visualize the distribution of petal length/width for each species
df.boxplot(column="petal_length", by="species")

plt.show()

df.boxplot(column="petal_width", by="species")
plt.show()

# Calculate the correlation matrix for the numeric features in the dataset
numeric_df = df.drop(columns=["species"])

corr = numeric_df.corr()
print("\nCorrelation matrix:")
print(corr)

# Visualize the correlation matrix using a heatmap
plt.imshow(corr)
plt.colorbar()

plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)

plt.show()

# Define features and target variable

X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = df["species"]

# Split the data into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set

from sklearn.metrics import accuracy_score

predictions = model.predict(X_test)

# Evaluate the model's accuracy

accuracy = accuracy_score(y_test, predictions)
print("\nModel Accuracy:")
print(f"Accuracy: {accuracy:.2f}")

# Create a confusion matrix to evaluate the model's performance in more detail
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predictions)

print(cm)