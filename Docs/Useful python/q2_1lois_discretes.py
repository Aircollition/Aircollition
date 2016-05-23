# -*- coding: utf-8 -*-
import numpy as np #import de numpy sous l'alias np
import numpy.random as npr #import de numpy.random sous l'alias npr
import matplotlib.pyplot as plt #import de matplotlib.pyplot sous l'alias plt
###############################################################################
## Question 2.1
###############################################################################
## Pour cette question, donnons deux facons de faire:
##  1. Une methode "a la main" ou on part de tirages de loi uniforme sur [0, 1]
##  2. Une fonction toute faite de numpy
# Nombre de simulations
n = 1000
# Methode 1. On utilise la fonction numpy.random.rand pour effectuer le tirage
# d'un echantillon X de taille n, suivant une loi uniforme sur [0,1]
X = npr.rand(n)
# On construit le vecteur M1 de taille X,tel que, quelque soit i 
# M1(i) = 1 si X(i) <= 0.3
# M1(i) = 2 si 0.3 < X(i) <= 0.9
# M1(i) = 3 si X(i) > 0.9
M1 = 1 * (X <= 0.3) + 2 * np.logical_and(0.3 < X, X <= 0.9) + 3 * (X > 0.9)
# Methode 2 : on utilise la fonction choice de numpy.random qui genere directement
# un echantillon de taille n, prenant les valeurs donnees en premier parametre
# selon les probabilites donnees par le parametre p
valeurs = np.array([1, 2, 3])
probas = np.array([0.3, 0.6, 0.1])
M2 = npr.choice(valeurs, size=n, p=probas)
#
# On dessine les resultats
# Remarque : la fonction hist de matplotlib n'est pas tres appropriee pour
#            afficher l'histogramme d'une loi discrete. On va faire autrement.
# On compte le nombre de fois que l'on voit 1, 2, 3 dans le vecteur
#
# numpy.bincount compte, dans un tableau d'entiers positifs ou nuls,
# le nombre d'elements du vecteur egaux a 0, 1, 2, etc.
# On cree donc le vecteur count1, via np.bincount(M1), dont on enleve la premiere valeur 
# qui traduit que l'on observe 0 fois 0, d'ou count1 = np.bincount(M1)[1:], idem pour
# la seconde methode avec le vecteur M2
counts1 = np.bincount(M1)[1:]
counts2 = np.bincount(M2)[1:]
# On convertit le vecteur counts1 en float pour pouvoir diviser par n 
counts1 = np.array(counts1, dtype=float)
counts1 /= n
# On peut aussi diviser par float(n) pour ne pas faire une division entiere 
counts2 = np.array(counts2, dtype=float)
counts2 /= n
#
# Affichage des resultats
print "-" * 40
print "Question 2.1"
print "-" * 40
# On affiche l'histogramme
plt.bar(valeurs - 0.5, counts1, width=1., label=u"loi empirique, méthode 1",edgecolor='r',color='',linewidth=3)
plt.bar(valeurs - 0.5, counts2, width=1., label=u"loi empirique, méthode 2",color=[.5,.5,.5]) #codage rgb de la couleur des barres
# On affiche la loi theorique pour comparer
plt.stem(valeurs, probas, label=u"loi théorique",linefmt='r-',markerfmt='r*',basefmt='b-')
# On dessine la legende
plt.legend(loc='upper left' , bbox_to_anchor = (1., 1.)) #on place la legende en haut a droite de la figure
# On donne un titre
plt.title(u"Question 2.1 : Loi discrète")
# On affiche toutes les figures
plt.show()
