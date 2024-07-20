import numpy as np


class GenerateRegression:
    def __init__(self):
        pass

    def generateDegree(self, degree, domain, n_samples, bias=0, noise=3, n_informative=0, isReplicable=False,
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

        start_point, end_point = domain

        x = np.linspace(start_point, end_point, n_samples)
        if noise != 0:
            x = x + np.random.normal(loc=0, scale=abs((end_point - start_point) / n_samples), size=x.shape)
            x[-1] = end_point

        coefficients = np.random.uniform(low=-1, high=1, size=degree + 1) / 2
        coefficients = np.sort(coefficients)
        coefficients[0] = bias

        X_eq = self.generateDesingMatrix(x, degree)

        Y_means = X_eq.dot(coefficients)
        y = np.random.normal(loc=Y_means, scale=noise, size=n_samples)

        if 0 < n_informative <= n_samples:
            indexes = np.random.choice(range(n_samples), n_informative, replace=False)
            y[indexes] = Y_means[indexes]

        return x, y

    def generateDesingMatrix(self, x_vector, degree):
        """
        Generates a design matrix for polynomial regression.

        Parameters:
        x_vector (np.ndarray): Array of x values.
        degree (int): Degree of the polynomial.

        Returns:
        np.ndarray: Design matrix for polynomial regression.
        """
        n_samples = x_vector.shape[0]
        designMatrix = np.ones((n_samples, degree + 1))
        for i in range(n_samples):
            for d in range(1, degree + 1):
                designMatrix[i, d] = np.power(x_vector[i], d)

        return designMatrix

    def generateRandom(self, domain, n_samples, bias=0, noise=3, isReplicable=False, seed=0):
        """
        Generates random data with a specified trend and noise.
        Take the last y and generate a new y based on the normal distribution.

        Parameters:
        domain (tuple): Range of x values (start, end).
        n_samples (int): Number of samples.
        bias (float): Starting value of y.
        noise (float): Standard deviation of the Gaussian noise.
        isReplicable (bool): Whether to use a fixed seed for reproducibility.
        seed (int): Seed for the random number generator.

        Returns:
        tuple: Arrays of x and y values.
        """

        if isReplicable:
            seed_ = int(bias * 2 + noise * 3 + seed * 5 + n_samples * 7 + seed * 11)
            np.random.seed(seed_)

        start_point, end_point = domain
        x = np.linspace(start_point, end_point, n_samples)

        y = np.array([bias])
        for i in range(n_samples - 1):
            y = np.append(y, (np.random.normal(loc=y[-1], scale=noise, size=1)))

        return x, y
