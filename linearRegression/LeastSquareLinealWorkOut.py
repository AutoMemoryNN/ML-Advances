import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
import sklearn

x, y = make_regression(
    n_samples=100,
    n_features=1,
    n_informative=2,
    bias=-10,
    noise=11,
    random_state=25
)

y = y.reshape(-1, 1)

x_b = np.c_[np.ones((x.shape[0], 1)), x]

# compute la Moore-Penrose pseudo-inverse
XtX = x_b.T.dot(x_b)  # (X^T).dot(X)
invXtX = np.linalg.inv(XtX)  # (X^T X)^-1
pseudoInvMoorePenrose = invXtX.dot(x_b.T)  # (X^T X)^-1 X^T

# Compute Normal Equation
bestW = pseudoInvMoorePenrose.dot(y)
print("Coefficients:", bestW)

plt.subplots(figsize=(8, 5))
plt.scatter(x, y, marker='o', label='Original Data')

X_new = np.linspace(min(x), max(x), 100).reshape(-1, 1)

y_pred = X_new.dot(bestW[1]) + bestW[0]

# Dibujar la línea de regresión
plt.plot(X_new, y_pred, color='red', linewidth=2, label='Línea de regresión')
plt.xlabel("Característica")
plt.ylabel("Objetivo")
plt.legend()
plt.show()