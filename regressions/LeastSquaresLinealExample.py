import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression

X, y = make_regression(
    n_samples=100,
    n_features=2,
    n_informative=2,
    noise=10,
    random_state=25
)

# concatenate a colum fulfill of ones
X_b = np.concatenate([np.ones((len(X), 1)), X], axis=1)

# Compute the Normal Equation
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
# for find the best values for the vector w
intercept, *coef = theta_best
print(f"Intercept: {intercept}\nCoeficients: {coef}")

# Display feature1 vs target
plt.subplots(figsize=(8, 5))
plt.scatter(X[:, 1], y, marker='o', label='Datos originales')

# Compute the predicted values
X_range = np.linspace(min(X[:, 1]) - 1, max(X[:, 1] + 1), 100)
X_range_b = np.c_[np.ones((100, 1)), np.zeros((100, 1)), X_range]
y_pred = X_range_b.dot(theta_best)

# Drawn
plt.plot(X_range, y_pred, color='red', linewidth=2, label='Línea de regresión')
plt.xlabel("Feature at Index 1")
plt.ylabel("Target")
plt.legend()
plt.show()
