# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr
import scipy.stats as sps

#nombre de tirages
N = 10**6
# Esperance
m = 10.
# Ecart-type
s = 3.
#tirages
X = m + s * npr.randn(N)
x = np.linspace(min(X), max(X), 100)
#densite
f_x = sps.norm.pdf(x,m,s)
#figure()
plt.hist(X, normed=True, label="Histogramme")
plt.plot(x, f_x, "r", label=u"Densité  \n de probabilité")# Noter le retour a la ligne dans la legende
plt.legend(loc=2)
plt.title(u"Loi gaussienne d'espérance " + str(m) + u" et d'écart-type " + str(s))
plt.show()

