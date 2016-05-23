# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr
#
###############################################################################
## Loi des grands nombres pour la loi uniforme sur [0,1]
N = 10**5 # taille de l'echantillon
X = npr.rand(N) # tirage de N valeurs, stockees dans X, suivant la loi uniforme sur [0,1]
M = np.cumsum(X) / np.arange(1, N+1) # somme cumulee dans M des valeurs de X, divisee respectivement par 1, 2, 3, ... N
plt.figure()
p, = plt.plot(M)
a = plt.axhline(0.5, color="r") #tracer d'une droite horizontale passant en 0.5, milieu de [0,1]
plt.legend([p, a], ["$S_n / n$", "$E(X)$"])
plt.title("Loi des grands nombres pour la loi uniforme sur [0, 1]")
plt.axis([0,N,.45,.55])
#
###############################################################################
## Loi des grands nombres pour la loi normale de moyenne m et d'ecart type s 
m=1
s=2
X = m+ s*npr.randn(N)
M = np.cumsum(X) / np.arange(1, N+1)
plt.figure()
p, = plt.plot(M)
a = plt.axhline(m, color="r") #tracer d'une droite horizontale passant en m, moyenne de la gaussienne
plt.legend([p, a], ["$S_n / n$", "$E(X)$"])
plt.title("Loi des grands nombres pour la loi gaussienne \n de moyenne $m =$ " +str(m) +u" et d'écart type $s =$ " +str(s))
plt.axis([0,N,.5,1.5])
plt.show()
