""" Linear Regression Exercise

generate fake linear data
fit a linear regression
plot line """

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = 2 * np.random.rand(100,1)
y = 4 + 3*X + np.random.rand(100,1)

lin_reg = LinearRegression()
lin_reg.fit(X,y)

plt.scatter(X,y)
plt.plot(X, lin_reg.predict(X), color='red')
plt.title('Linear Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.show()