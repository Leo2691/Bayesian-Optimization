import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def add_powers(data, n=5):
    result = [data]
    for i in range(2, n + 1):
        poly = np.power(data, i)
        result.append(poly)
    return np.vstack(result).T


def generate_data(n=20):
    x = np.linspace(-10, 10, 100)
    y = x * np.cos(x)
    max_y = np.max(y)
    y += np.random.random(y.shape)
    idx = np.random.choice(len(y), n)
    return x[idx], y[idx], max_y



original_data, target, max_y = generate_data(n=10)

power = 5

data = add_powers(original_data, power)
max_vals = np.array([max_y ** i for i in range(1, power + 1)])
data /= max_vals

"""w = (np.random.random(power) - 0.5) * 1.
b = -1.
learning_rate = 0.1

for e in range(100000):
    h = data.dot(w) + b
    w -= learning_rate * ((h - target)[:, None] * data).mean(axis=0)
    b -= learning_rate * (h - target).mean()"""

#plt.title('Interpolating f(x) = x * cos(x)')
#x = add_powers(np.linspace(-10, 10, 100), power)
#x /= max_vals
#y = x.dot(w) + b
#plt.plot(x[:, 0] * max_vals[0], y, lw=3, c='g')
#plt.scatter(data[:, 0] * max_vals[0], target, s=50)
#plt.show()

## Bayesian way

from itertools import product

def likelihood(x, mu, sigma=1.):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))

frequency = 5
posterior = np.ones(frequency ** (power + 1)) / float(frequency)
param_values = [np.linspace(-10, 10, frequency) for _ in range(power + 1)]

for i, pt in enumerate(data):
    for j, params in enumerate(product(*param_values)):
        b = params[0]
        w = np.array(params[1:])
        h = pt.dot(w) + b
        like = likelihood(pt[0], h)
        posterior[j] *= like
    posterior /= posterior.sum()

a = 0;