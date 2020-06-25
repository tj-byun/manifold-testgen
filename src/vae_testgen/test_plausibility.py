import numpy as np
from scipy.stats import norm
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')

P_STANDARD_NORMAL = 0.3989422804014327


def f1(x):
    x1 = np.array([norm(0, 1).pdf(xi) for xi in x])
    return 1.0 / np.power(np.log2(x1 / P_STANDARD_NORMAL) - 1, 2)


def f2(x):
    return 2.0 / np.power(np.e, 0.5 * x) - 1.0


def f3(x):
    return 1.0 / np.power(np.e, 0.5 * np.power(x, 2))


def f4(x):
    return (1.0 / np.power(np.e, 0.5 * np.power(x, 2))) ** 2


x = np.linspace(0, 3, num=100)
print(x)
y1, y2, y3, y4 = f1(x), f2(x), f3(x), f4(x)

plt.plot(x, y3)
plt.savefig('ys.png')
plt.show()
plt.plot(x, y2)
plt.plot(x, y3)
plt.plot(x, y4)
