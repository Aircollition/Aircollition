# Approximation de la constante d'Euler, via une boucle 
# VS via l'utilisation des fonctions cablees
#
import numpy as np
import time
#import time pour avoir la fonction qui donne le temps CPU
# different de la fonction time de base, qui donne l'heure

n = 10**6
# Methode 1. Boucle for
print "-" * 40
print "Methode 1. Boucle for"
print "-" * 40
t1 = time.time()
x = 0
for i in range(1, n+1):
    x += 1. / i
print "gamma=", np.sum(x) - np.log(n)
t2 = time.time()
print "Cela a pris ", t2 - t1, " secondes"
# Methode 2. Numpy, programmation vectorielle
print "-" * 40
print "Methode 2. Numpy, programmation vectorielle"
t1 = time.time()
print "gamma=", np.sum(1. / np.arange(1, n+1)) - np.log(n)
t2 = time.time()
print "Cela a pris ", t2 - t1, " secondes"
print "-" * 40
# Exemple de resultats
#----------------------------------------
#Methode 1. Boucle for
#----------------------------------------
#gamma= 0.577216164901
#Cela a pris  0.217562913895  secondes
#----------------------------------------
#Methode 2. Numpy, programmation vectorielle
#gamma= 0.577216164901
#Cela a pris  0.0111699104309  secondes
#----------------------------------------#
