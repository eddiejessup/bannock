from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt


def find_c(rho,
           dx, dt,
           alpha, delta, D):
    c = np.zeros_like(rho)
    laplace_c = np.zeros_like(c)
    for _ in range(100000):
        laplace_c[1:-1] = (c[2:] + c[:-2] - 2.0 * c[1:-1]) / dx ** 2
        laplace_c[0] = 0.0
        laplace_c[-1] = 0.0
        c += (D * laplace_c + alpha * rho - delta * c) * dt
    return c

x = np.linspace(-2000.0, 2000.0, 800)
dx = x[1] - x[0]
delta = 0.01
alpha = 1.0
D = 100.0
d = 10.0 * np.exp(-0.01 * np.abs(x))

c = np.zeros_like(d)
for i in range(len(x)):
    for j in range(len(x)):
        c[i] += d[j] * np.exp(-np.sqrt(delta / D) * np.abs(x[i] - x[j]))
c *= 0.5 * dx

cs = find_c(d, dx=dx, dt=0.1, alpha=alpha, delta=delta, D=D)

plt.plot(x, d, label='density')
plt.plot(x, cs, label='c direct')
plt.plot(x, c, label='c analytic')

plt.legend()
plt.show()
