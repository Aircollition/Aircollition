# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr
import scipy.stats as sps

#nombre de tirages
N = 5000
#parametre de la loi gamma, k=shape et theta=scale
shape= 8.
scale= 4.
#tirages
X = npr.gamma(shape, scale,size= N)
#densite
x = np.linspace(min(X), max(X), 100)
f_x = sps.gamma.pdf(x, shape, scale=scale)
#figure()
plt.hist(X, normed=True, label="Histogramme")
plt.plot(x, f_x, "r", label=u"Densité de probabilité")
plt.legend()
plt.title("Loi $gamma$ de forme " + str(shape) + u" et d'échelle " + str(scale))
plt.show()
