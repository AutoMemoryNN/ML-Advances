import numpy as np

class GenerateRegression:
    def __init__(self):
        pass

    def generateDegree(self, degree, polyRange, n_samples, bias=0, noise=3, isReplicable=False, n_informative=0,
                       seed=0):
        """
        Generates polynomial regression data.

        Parameters:
        degree (int): Degree of the polynomial.
        polyRange (tuple): Range of x values (start, end).
        n_samples (int): Number of samples.
        bias (float): Bias term in the polynomial.
        noise (float): Standard deviation of the Gaussian noise.
        isReplicable (bool): Whether to use a fixed seed for reproducibility.
        n_informative (int): Number of informative samples.
        seed (int): Seed for the random number generator.

        Returns:
        tuple: Arrays of x and y values.
        """

        if isReplicable:
            seed_ = int(degree * 2 + n_samples * 3 + bias * 5 + noise * 7 + n_informative * 11 + seed * 13)
            np.random.seed(seed_)

        start_point, end_point = polyRange
        X = np.linspace(start_point, end_point, n_samples)

        coefficients = np.random.uniform(low=-2, high=2, size=degree + 1)
        coefficients = np.sort(coefficients)
        coefficients[0] = bias

        X_eq = np.ones((n_samples, degree + 1))
        for i in range(n_samples):
            for d in range(1, degree + 1):
                X_eq[i, d] = np.power(X[i], d)

        Y_means = X_eq.dot(coefficients)
        Y = np.random.normal(loc=Y_means, scale=noise, size=n_samples)

        if 0 < n_informative <= n_samples:
            indexes = np.random.choice(range(n_samples), n_informative, replace=False)
            Y[indexes] = Y_means[indexes]

        return X, Y

