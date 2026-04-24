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

poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
