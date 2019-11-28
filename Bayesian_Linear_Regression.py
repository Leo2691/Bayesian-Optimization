import numpy as np
import matplotlib.pyplot as plt

def generate_data(n=20, k = 2, b = 3):
    x = np.linspace(-10, 10, 100)
    y = k * x + b;
    max_y = np.max(y)
    y += np.random.random(y.shape) ** 2
    idx = np.random.choice(len(y), n)
    return x[idx], y[idx], max_y


x, y, max_y = generate_data(n=10);


#plt.title('Interpolating f(x) = x * cos(x)')
#x = add_powers(np.linspace(-10, 10, 100), power)
#x /= max_vals
#y = x.dot(w) + b
#plt.plot(x, y, lw=3, c='g')
#plt.scatter(x, y, s=50)
#plt.show()

from itertools import product

def likelihood(x, mu, sigma=1.):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))


frequency = 50
posterior = np.ones(frequency ** 2) / 2;
param_values = [np.linspace(-10, 10, frequency) for _ in range(2)]

comb_param = [np.zeros(2) for i in range(frequency ** 2)];
a = 0;

# cycle of objects
for i, pt in enumerate(x):
    # cycle of parameters combinations
    for j, params in enumerate(product(*param_values)):
        k = params[0]
        b = params[1]
        z = k * pt + b
        like = likelihood(y[i], z)
        if (like != 0):
            posterior[j] *= like
        comb_param[j] = params
    posterior /= posterior.sum()

#best_params = max(enumerate(posterior), key=lambda x: x[1])[0];
ind = np.array(posterior).argmax()
best_params = comb_param[ind];

print(best_params, ind)

comb_param = np.vstack(comb_param)
plt.plot(range(0, frequency ** 2)[1400:1500], posterior[1400:1500])
plt.show();

print(comb_param[1400:1500])

u = 1;