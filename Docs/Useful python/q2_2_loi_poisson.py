# -*- coding: utf-8 -*-
import numpy as np
import numpy.random as npr
import scipy.stats as sps
import matplotlib.pyplot as plt
###############################################################################
## Question 2.2.2
###############################################################################
# Intensite de la loi de poisson
mu = 0.5
# Nombre de simulations
N = 1000
# Tirage d'un echantillon de loi de Poisson, de taille N
X = npr.poisson(mu, N)
n = 20
# numpy.bincount compte, dans un tableau d'entiers positifs ou nuls,
# le nombre d'elements du vecteur egaux a 0, 1, 2, etc.
# on divise le resultat par float(N), pour ne pas faire une division entiere
#
counts = np.bincount(X) / float(N)
#
# Discretisation de la loi theorique
#
x = np.arange(len(counts))
f_x = sps.poisson.pmf(x, mu)
#
# Affichage compare, empirique vs theorique
#
plt.bar(x - 0.5, counts, width=1., label="loi empirique",edgecolor=[.6,.6,.6],color='') 
# Nota  ; edgecolor=[.6,.6,.6],color='' => le contour des barres sera gris, les barres ne seront pas remplies
p2 = plt.stem(x, f_x, "r", label=u"loi théorique")
plt.title("Question 2.2.2 : loi de Poisson")
plt.legend()
plt.show
