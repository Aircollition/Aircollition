# -*- coding: utf-8 -*-
from pylab import pi
from time import time
import numpy as np #import de numpy sous l'alias np
import numpy.random as npr #import de numpy.random sous l'alias npr
import matplotlib.pyplot as plt #import de matplotlib.pyplot sous l'alias plt

#Nombre de tirages
N = 10**5
#on genere dans X des realisations de la loi de Wigner par methode du rejet
borne = 1. / pi
temps_debut = time()
u=4*npr.rand(N)-2 # u = tirage de N valeurs suivant une loi uniforme sur [-2,2]
v=borne*npr.rand(N) # v =tirage de N valeurs suivant une loi uniforme sur [0,1/pi]
# On garde dans X tous les éléments de u vérifiant v(i)<sqrt(4*u(i)^2)/(2*pi)
X=u[(v<np.sqrt(4.-u**2)/(2*pi))]
temps_calcul_vectoriel = time()-temps_debut
# On récupère dans N_X la taille du vecteur X ainsi construit
N_X = np.size(X)
#densite de la loi de Wigner
x = np.linspace(-2., 2., 100) # discrétisation de l'intervalle [-2,2]
f_x = 1 / (2 * pi) * np.sqrt(4 - x ** 2) # discrétisation de la densité théorique de Wigner
# Affichage des resultats
plt.hist(X, normed=True, bins=round(np.sqrt(N_X)/10), label="Loi de Wigner empirique")
plt.plot(x, f_x, "r", label=u"Densité de la loi de Wigner :  $f(x)=2*\pi/\sqrt{4-x^2}$")
plt.legend(loc='upper left' , bbox_to_anchor = (1., 1.))
plt.show()
print u"Temps de calcul de ", N_X, u" réalisations de la loi de Wigner, en vectoriel = ",temps_calcul_vectoriel
#
# Calcul via des boucles
#Nombre de tirages
N = N_X
#on genere dans X N_X realisations de la loi de Wigner par methode du rejet
#on comparera le temps de calcul avec des boucles avec le temps de calcul en vectoriel
temps_debut = time()
X = []
k = 0
while k < N:
    u = 4. * npr.rand() - 2.
    v = borne * npr.rand()
    while v > np.sqrt(4. - u ** 2) / (2 *pi):
        u = 4 * npr.rand() - 2
        v = borne * npr.rand()
    X.append(u)
    k += 1
    # fin du tant que
# fin de tant que
temps_calcul_boucle = time()-temps_debut
#densite de la loi de Wigner
x = np.linspace(-2., 2., 100)
f_x = 1 / (2 * pi) * np.sqrt(4 - x ** 2)
#
plt.hist(X, normed=True, bins=round(np.sqrt(N)/10), label="Loi de Wigner empirique")
plt.plot(x, f_x, "r", label=u"Densité de la loi de Wigner")
plt.legend(loc='lower right')
plt.show()
print u"Temps de calcul de ", N_X, u" réalisations de la loi de Wigner, via des boucles = ",temps_calcul_boucle
# Calcul du gain de temps
gain = (temps_calcul_boucle-temps_calcul_vectoriel)/temps_calcul_boucle*100
print "gain du vectoriel sur les boucles = ",gain,"%"
#Exemple de resultats
# Temps de calcul de  78604  réalisations de la loi de Wigner, en vectoriel =  0.00872087478638
# Temps de calcul de  78604  réalisations de la loi de Wigner, via des boucles =  0.418179035187
# gain du vectoriel sur les boucles =  97.9145595421 %
