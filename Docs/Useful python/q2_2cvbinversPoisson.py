# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
###############################################################################
## Question 2.2.3
###############################################################################
n = 1000
p = 0.005
# Intensite de la loi de poisson
mu = p * n
np2 = n * p ** 2
x = np.arange(0, 7 * mu)
# Simulation de la loi de Poisson
p_poisson = sps.poisson.pmf(x, mu)
# Simulation de la loi Binomiale
p_binom = sps.binom.pmf(x, n, p) 
#
#Affichage compare des resultats
plt.stem(x, p_poisson, markerfmt='*',label="poisson")
#On shifte de 0.3 en abscisse les resultats obtenus pour la loi binomiale
plt.stem(x+0.3, p_binom, "r", label="binomiale")
plt.title("Question 2.2.3:  limite loi binomiale vers loi de Poisson avec $np^2=$"+str(np2))
plt.legend()
plt.show()
