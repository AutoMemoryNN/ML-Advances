import matplotlib.pyplot as plt
import numpy as np
import GenerateRegression as gen

generator = gen.GenerateRegression()


def gaussianBasicFunction(x, s, miu):
    return np.exp(-(((x - miu) / (2 * s)) ** 2))


def linearRegression(x, y, degree=1, regularizationCoefficient=0.0, regressionType='Lineal'):
    y = y.reshape(-1, 1)

    best_coefficients = None
    x_b = x_b = generator.generateDesingMatrix(x, 1)

    if regressionType == 'Polynomial':
        x_b = generator.generateDesingMatrix(x, degree)
    elif regressionType == 'Gaussian':
        x_b = np.array([gaussianBasicFunction(x_i, 0.01, x_i) for x_i in x])[:,
              np.newaxis]  # BAD, where can i get miu and s?

    if regularizationCoefficient != 0.0:
        XtX = x_b.T.dot(x_b)  # (X^T).dot(X)
        regularizationCoefficientI = regularizationCoefficient * (np.identity(x_b.shape[1]))  # λ I
        invXtXRegularized = np.linalg.inv(np.add(XtX, regularizationCoefficientI))  # (X^T X + λ I)^-1
        pseudoInvMoorePenroseFitted = invXtXRegularized.dot(x_b.T)  # (X^T X + λ I)^-1 X^T
        best_coefficients = pseudoInvMoorePenroseFitted.dot(y)

    else:
        # compute Moore-Penrose pseudo-inverse
        XtX = x_b.T.dot(x_b)  # (X^T).dot(X)
        invXtX = np.linalg.inv(XtX)  # (X^T X)^-1
        pseudoInvMoorePenrose = invXtX.dot(x_b.T)  # (X^T X)^-1 X^T
        best_coefficients = pseudoInvMoorePenrose.dot(y)

    x_new = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)

    x_new_b = None
    if regressionType == 'Polynomial':
        x_new_b = generator.generateDesingMatrix(x_new, degree)
    elif regressionType == 'Gaussian':
        x_new_b = np.array([gaussianBasicFunction(x_i[0], 0.01, x_i[0]) for x_i in x_new])[:,
                  np.newaxis]  # BAD, where can i get miu and s?

    y_predict = x_new_b.dot(best_coefficients)

    return x_new, y_predict, best_coefficients


def polynomialRegression(x, y, degree, regularizationCoefficient=0.0):
    return linearRegression(x, y, degree, regularizationCoefficient, regressionType='Polynomial')


def gaussianRegression(x, y, degree, regularizationCoefficient=0.0):
    return linearRegression(x, y, degree, regularizationCoefficient, regressionType='Gaussian')


def errorFunction(y_true, y_pred):
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true {y_true.shape} and y_pred {y_pred.shape} must have the same shape")
    return np.log(0.5 * np.sum((y_true - y_pred) ** 2))  # natural logarithm for the error


def evaluatePolynomial(x, coefficients):
    x = np.asarray(x)
    x_eq = generator.generateDesingMatrix(x, len(coefficients) - 1)
    return x_eq.dot(coefficients)


def findBestPolynomialRegression(x, y, regularizationCoefficient=0.0):
    errors = np.zeros((30,))  # TODO : max degree for seeking was set at 30
    for i in range(30):
        _, _, co = polynomialRegression(x, y, i, regularizationCoefficient)
        errors[i] = errorFunction(y.reshape(-1, 1), evaluatePolynomial(x, co))

    min_error = np.min(errors)
    best_degree = np.argmin(errors)
    return min_error, best_degree


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
    # x, y = generator.generateDegree(3, (-5, 5), 100, 10, 1, 7,
    #                                True, 2)
    x, y = generator.generateRandom((-20, 20), 100, 10, 0.5, True, 2)

    initializePlot('Polynomial Regression')
    plotData(x, y, 'Original Data', isScatter=True, color='blue')

    err, degree = findBestPolynomialRegression(x, y, -3)

    print(f'Error: {err}', degree)

    x_new, y_predict, y_predict_co = polynomialRegression(x, y, degree)
    plotData(x_new, y_predict, 'Regression Line', isLine=True, color='red')

    plt.text(2, 2, err)

    plt.show()


if __name__ == '__main__':
    main()
