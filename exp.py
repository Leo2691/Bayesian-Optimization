import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal

# Network params
x = np.array([[.2, .3, .4],
              [.3, .2, .3],
              [.1, .2, .3],
              [.4, .5, .1]
              ])
#Network Score
er = np.array([5, 4, 3, 2])

mu = np.zeros(x.shape[1], dtype=float)
#for i in np.arange(x.shape[1]):
    #mu[i] = np.mean(x[:, i])

#params of prior destribution
mu = np.mean(x, axis=1)
Sig = x.T.dot(x)
cov = np.cov(x)

u = np.linalg.inv(Sig)

# Output block----------------------------------------
"""r = multivariate_normal.rvs(mean=mu, cov=Sig, size=100)
print(mu)
print(Sig)
print(cov)
plt.scatter(r[:, 0], r[:, 1])
plt.show()"""

x_ = np.array([.1, .3, .2])

cov1 = x_.T.dot(x_)
Sig1 = x_.dot(x.T)
cov2 = np.cov(x_, x.T)


print(cov1)
