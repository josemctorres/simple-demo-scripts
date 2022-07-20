import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal
from sklearn.linear_model import LinearRegression

print("Linear Regression sklearn example")
# Dataset generation: N values for X between 0 and A
N = 200
A = 30
X = A * np.random.random((N, 1))
theta_0, theta_1 = 10, 2.5
print("true theta0 =", theta_0, "; true theta1 =", theta_1)
mu, sigma = 0, 5
y = theta_1 * X + theta_0 + normal(loc=mu, scale=sigma, size=X.shape)  # y = a*x + b + gaussian_noise
model = LinearRegression()
model.fit(X, y)
x_pred = [[np.min(X)], [np.max(X)]]
y_pred = model.predict(x_pred)  # predict y from the data
plt.scatter(X, y)
plt.title("sklearn linear regression example")
plt.axline((np.min(X), y_pred[0][0]), (np.max(X), y_pred[1][0]), linewidth=1, color='r')
plt.show()
