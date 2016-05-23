# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr
import scipy.stats as sps

#Nombre de tirages de Monte Carlo pour calculer S_N
N = 100000
#Nombre de realisations de sqrt(N)S_N
n=1000
# Lois uniformes sur [-sqrt(3), sqrt(3) ]
X = 2 * np.sqrt(3) * (npr.rand(n,N) - 0.5) # Tirage dans la matrice X de (nxN) valeurs suivant une loi uniforme sur [-sqrt(3),sqrt(3)]
m = 0.
s = 1.
T = np.sqrt(N) * np.mean(X, axis=1) # Calcul dans T, des moyennes des lignes de la matrice X, divisees par sqrt(n)
borne = max(abs(min(T)),max(T)) # Calcul dans borne de la valeur absolue du max des valeurs de T
x = np.linspace(-borne, borne, 100) # discretisation dans x du segment [-borne,borne]
f_x = sps.norm.pdf(x, m, s) # calcul de la gaussienne standard en x
#
# Afficahge des resultats
#
plt.figure()
plt.hist(T, normed=True, bins=30, label="histogramme")
plt.plot(x, f_x, "r", label=u"Densité de la loi gaussienne standard")
plt.legend(loc='upper left' , bbox_to_anchor = (0., -0.1))
plt.title(u"Théorème central limite pour les lois uniformes sur [$-\sqrt{3}$, $\sqrt{3}$]")
#
#on recupere la 1ere trajectoire de X dans X1
#
X1 = X[0, :]
arange_N = np.arange(1,N+1)
T1 = np.sqrt(arange_N) * np.cumsum(X1) / arange_N # Calcul dans T1 des sommes cumulees de X1, normalisees respectivement par 1, 2, ...,N
#
# Affichage de T1
#
plt.figure()
plt.plot(T1)
plt.title("Evolution de $\sqrt{N}(\overline{X} - m)$ en fonction de N")
plt.show()
