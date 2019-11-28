from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

import numpy as np


def vector_2d(array):
    return np.array(array).reshape((-1, 1))


def EI_GP(x_train, y_train, x_test, scores, param_choices):
                # param_choices and x_test are same.
                # y_train and scores are same

                kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
                gp = GaussianProcessRegressor(kernel, n_restarts_optimizer=100, normalize_y=True)
                gp.fit(x_train, y_train)

                # Get mean and standard deviation for each possible
                # number of hidden units
                y_mean, y_std = gp.predict(x_test, return_std=True)
                y_std = vector_2d(y_std)

                y_min = min(scores)  # np.min(scores, axis=1)

                # Calculate expected improvement from 95% confidence interval
                expected_improvement = y_min - (y_mean - 1.96 * y_std)
                expected_improvement[expected_improvement < 0] = 0

                max_index = expected_improvement.argmax()
                # Select next che based on expected improvement
                new_param = param_choices[max_index]


# Network params
x = np.array([[2, 3, 4],
              [3, 2, 3],
              [1, 2, 3],
              [4, 5, 1]
              ])
#Network Score
er = np.array([5, 4, 3, 2])

x_test = np.array([np.arange(10) for i in np.arange(3)]).T

EI_GP(x_train=x, y_train=er, x_test=x_test, scores=er, param_choices=x_test)

