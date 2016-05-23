# -*- coding: utf-8 -*-
import numpy as np
import numpy.random as npr
import scipy.stats as sps
import matplotlib.pyplot as plt
###############################################################################
## Question 2.2.1
###############################################################################
#Nombre de tirages
N = 10000
#Parametres de la binomiales
n = 20
p = 0.3
#N tirages de la binomiale
B = npr.binomial(n, p, N)
#Poids de la binomiale aux points 0, ...., n
valeurs = np.arange(n+1)
f = sps.binom.pmf(valeurs, n, p)


#Histogramme de la distribution empirique avec bincount et bar
partial_counts = np.bincount(B)
#on complete avec des zeros le vecteur de comptage en un vecteur de meme taille que valeur 
counts = np.zeros(n+1)
counts[0:len(partial_counts)] = partial_counts 
counts = np.array(counts, dtype=float) # on transforme le vecteur d'entiers en vecteur de flotants
counts /= N
plt.bar(valeurs - 0.5, counts, width=1., label="loi empirique",color=[.6,.6,.6]) # les barres seront grises
#
#distribution theorique
plt.title("Question 2.2.1 : loi binomiale")
plt.stem(valeurs, f, "r", label=u"loi théorique")
plt.legend()
plt.show()

