import matplotlib.pyplot as plt
import numpy as np
import GenerateRegression as gen


def polynomialRegression(x, y, degree):
    plt.subplots(figsize=(8, 5))
    plt.scatter(x, y, marker='o', label='Original Data')

    y = y.reshape(-1, 1)

    x_b = generator.generateDesingMatrix(x, degree)

    # compute Moore-Penrose pseudo-inverse
    XtX = x_b.T.dot(x_b)  # (X^T).dot(X)
    invXtX = np.linalg.inv(XtX)  # (X^T X)^-1
    pseudoInvMoorePenrose = invXtX.dot(x_b.T)  # (X^T X)^-1 X^T

    best_coefficients = pseudoInvMoorePenrose.dot(y)

    x_new = np.linspace(x.min() * 1.2, x.max() * 1.2, 100).reshape(-1, 1)

    x_new_b = generator.generateDesingMatrix(x_new, degree)

    y_predict = x_new_b.dot(best_coefficients)

    plt.plot(x_new, y_predict, color='red', linewidth=2, label='regression line')
    plt.xlabel("Feature")
    plt.ylabel("Objetive")
    plt.legend()
    plt.show()


generator = gen.GenerateRegression()
x, y = generator.generateDegree(2, (-10, 10), 75, 10, 4, 5, True, 1)
# x, y = generator.generateRandom((-20, 20), 100, 10, 0.5, True, 2)
polynomialRegression(x, y, 2)
