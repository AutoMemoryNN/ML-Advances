import numpy as np
import random as rd
import scipy as sp


class GenerateRegression:
    def __init__(self):
        pass

    def generateDegree(self, degree, polyRange, n_samples, bias, noise, isAlwaysRandom, n_informative):
        if isAlwaysRandom:
            seed = int(degree * 2 + polyRange[0] * 3 + polyRange[
                1] * 5 + n_samples * 7 + bias * 11 + noise * 13 + n_informative * 17)
            np.random.seed(seed)

        X = np.array([])
        Y = np.array([])

        startPoint = polyRange[0]
        endPoint = polyRange[1]
        jumps = (endPoint - startPoint) / n_samples

        for j in range(n_samples - 1):
            i = startPoint + (j * jumps)
            x = np.random.normal(loc=i, scale=jumps, size=1)

            X = np.append(X, x[0])

        X = np.append(X, endPoint)

        print(X)

        return X, Y
