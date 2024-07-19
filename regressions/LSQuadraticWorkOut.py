import matplotlib.pyplot as plt
import numpy as np
import GenerateRegression as gen

generator = gen.GenerateRegression()
x, y = generator.generateDegree(2, (-10, 10), 50, 10, 20, False, 0, 1)

plt.subplots(figsize=(8, 5))
plt.scatter(x, y, marker='o', label='Original Data')

y = y.reshape(-1, 1)

x_b = generator.generateDesingMatrix(x, 2)

# compute la Moore-Penrose pseudo-inverse
XtX = x_b.T.dot(x_b)  # (X^T).dot(X)
invXtX = np.linalg.inv(XtX)  # (X^T X)^-1
pseudoInvMoorePenrose = invXtX.dot(x_b.T)  # (X^T X)^-1 X^T

best_coefficients = pseudoInvMoorePenrose.dot(y)

X_new = np.linspace(min(x), max(x), 100).reshape(-1, 1)

y_pred = (X_new ** 2).dot(best_coefficients[2]) + X_new.dot(best_coefficients[1]) + best_coefficients[0]

plt.plot(X_new, y_pred, color='red', linewidth=2, label='Línea de regresión')
plt.xlabel("Característica")
plt.ylabel("Objetivo")
plt.legend()
plt.show()
