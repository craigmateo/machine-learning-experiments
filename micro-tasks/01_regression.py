import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures

import numpy as np

# Regression

# Generate synthetic data
X = 2 * np.random.rand(100,1)
y = 4 + 3 * X + np.random.rand(100,1)

# inititalize and train regression modal (typically uses "least squares")
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# sample point 
# print(X[50],lin_reg.predict(X)[50])

# plot 
plt.scatter(X, y)
plt.plot(X, lin_reg.predict(X), color='red')
plt.title("Linear Fit")
plt.xlabel('X')
plt.ylabel('y')
plt.show()


# add outliers
X_out = np.vstack([X, [[2.5]]])
y_out = np.vstack([y, [[40]]])  # Add a strong outlier

lin_reg_out = LinearRegression().fit(X_out, y_out)

# plot with outliers
plt.scatter(X_out, y_out)
plt.plot(X_out, lin_reg_out.predict(X_out), color='orange', label="With Outlier")
plt.plot(X, lin_reg.predict(X), color='red', label="Original")
plt.legend()
plt.title("Impact of Outliers")
plt.xlabel('X')
plt.ylabel('y')
plt.show()

# Polynomial Regression

from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Make fake nonlinear y
y = X**2 + np.random.randn(len(X), 1) * 0.1

# Scale X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Polynomial features
poly = PolynomialFeatures(degree=9)
X_poly = poly.fit_transform(X_scaled)

print(X_scaled[:3])
print(X_poly[:3])

# Fit model
lin_poly = LinearRegression().fit(X_poly, y)

# Plot
plt.scatter(X, y)

X_range = np.linspace(0, 2, 100).reshape(-1, 1)

# IMPORTANT: scale X_range first
X_range_scaled = scaler.transform(X_range)
X_range_poly = poly.transform(X_range_scaled)

plt.plot(X_range, lin_poly.predict(X_range_poly), color='green')
plt.title("Polynomial Regression")
plt.xlabel('X')
plt.ylabel('y')
plt.show()

# Regularization (Ridge vs Lasso)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso

X = np.random.rand(30, 1) * 5
y = X**2 + np.random.randn(30, 1) * 4.0

linear_poly = make_pipeline(
    PolynomialFeatures(degree=9),
    StandardScaler(),
    LinearRegression()
)

ridge_poly = make_pipeline(
    PolynomialFeatures(degree=9),
    StandardScaler(),
    Ridge(alpha=10)
)

lasso_poly = make_pipeline(
    PolynomialFeatures(degree=9),
    StandardScaler(),
    Lasso(alpha=0.05, max_iter=100000)
)

linear_poly.fit(X, y)
ridge_poly.fit(X, y)
lasso_poly.fit(X.ravel().reshape(-1, 1), y.ravel())

X_range = np.linspace(0, 5, 300).reshape(-1, 1)

plt.figure(figsize=(8, 5))
plt.scatter(X, y, label="Data")

plt.plot(X_range, linear_poly.predict(X_range), label="Linear degree=9")
plt.plot(X_range, ridge_poly.predict(X_range), label="Ridge", linestyle="--")
plt.plot(X_range, lasso_poly.predict(X_range), label="Lasso", linestyle="-.")

plt.legend()
plt.show()
