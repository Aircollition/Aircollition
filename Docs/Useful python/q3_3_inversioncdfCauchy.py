# -*- coding: utf-8 -*-
from pylab import pi # import de la constante pi
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr

#densite de la loi de Cauchy
def f(x, a):
    '''
    Calcul de la densite de la loi de Cauchy \n
    f(x,a) = a / (pi * (x^2) + a^2) \n
    x et a sont les parametres d'entree
    '''
    return a / (pi * (x ** 2 + a ** 2))
# fin definition de la fonction f
#inverse de la fct de repartition de la loi de Cauchy
def G(x, a):
    '''
    Calcul de l'inverse de la fonction repartition de la loi de Cauchy \n
    G(x,a) = a * arctan(pi*(x-0.5) \n
    x et a sont les parametres d'entree
    '''
    return a * np.tan(pi * (x - .5))
# fin definition de la fonction G
#nombre de tirages
N = 5000
#parametre de la Cauchy
a =0.1
# Borne superieure de l'intervalle de discretiation
bound=10*a
#Simulation d'un ecahntillon d'une loi uniforme sur [0,1], de taille N
U = npr.rand(N)
# Calcul de l'inverse de la fonction repartition de la loi de Cauchy pour chaque composante de U
X = G(U, a)
# Discretisation de l'intervalle [-bound,+bound]
x = np.linspace(-bound, bound, int(np.sqrt(N)))
#Calcul de la densite de la loi de Cauchy pour chaque composante de x
y = f(x, a)
# Affichage des resultats
plt.figure()
plt.hist(X, normed=True, bins=round(np.sqrt(N)), label="histogramme", range=(-bound, bound),edgecolor='b',color=[1,1,1])
legende1 = u"densité de Cauchy : "+r"$f(x,a) = a/(\pi(x^2+a^2))$"
plt.plot(x, y, "r", label=legende1) 
plt.legend(loc='upper left' , bbox_to_anchor = (0., -.1))
plt.title(u"Loi de Cauchy de paramètre a = "  + str(a))
plt.show()
