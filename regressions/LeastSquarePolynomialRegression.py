import matplotlib.pyplot as plt
import numpy as np
import GenerateRegression as gen

generator = gen.GenerateRegression()


def polynomialRegression(x, y, degree):
    y = y.reshape(-1, 1)

    x_b = generator.generateDesingMatrix(x, degree)

    # compute Moore-Penrose pseudo-inverse
    XtX = x_b.T.dot(x_b)  # (X^T).dot(X)
    invXtX = np.linalg.inv(XtX)  # (X^T X)^-1
    pseudoInvMoorePenrose = invXtX.dot(x_b.T)  # (X^T X)^-1 X^T

    best_coefficients = pseudoInvMoorePenrose.dot(y)

    x_new = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)

    x_new_b = generator.generateDesingMatrix(x_new, degree)

    y_predict = x_new_b.dot(best_coefficients)

    return x_new, y_predict, best_coefficients


def initializePlot(title, figureSize=(8, 5), xLabel='Feature', yLabel='Objective'):
    plt.figure(figsize=figureSize)
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)


def plotData(x, y, label, isScatter=False, isLine=False, color=None):
    if isScatter:
        plt.scatter(x, y, marker='o', label=label, color=color)
    elif isLine:
        plt.plot(x, y, linestyle='-', label=label, color=color)
    plt.legend()


def main():
    x, y, co = generator.generateDegree(5, (-6, 6), 75, 10, 5, 1, True, 1)
    # x, y = generator.generateRandom((-20, 20), 100, 10, 0.5, True, 2)

    initializePlot('Polynomial Regression')
    plotData(x, y, 'Original Data', isScatter=True, color='blue')

    x_new, y_predict, _ = polynomialRegression(x, y, 5)
    plotData(x_new, y_predict, 'Regression Line', isLine=True, color='red')

    plt.show()


if __name__ == '__main__':
    main()
