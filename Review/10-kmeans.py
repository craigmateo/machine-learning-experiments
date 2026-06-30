from sklearn.datasets import load_wine
import pandas as pd

wine = load_wine(as_frame=True)
df = wine.frame

# 
X = df[["alcohol", "color_intensity"]]

# 
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 
from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

kmeans.fit(X_scaled)

df["cluster"] = kmeans.labels_

print(df[["alcohol", "color_intensity", "target", "cluster"]].head())

#

import matplotlib.pyplot as plt

plt.scatter(
    df["alcohol"],
    df["color_intensity"],
    c=df["cluster"]
)

plt.xlabel("Alcohol")
plt.ylabel("Color Intensity")
plt.title("K-Means clusters")
plt.show()