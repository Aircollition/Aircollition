# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 13:39:56 2016

@author: felipe garcia
"""
import numpy as np
from time import time
import numpy.linalg as npl

# Definition parametres modele
t1 = time()
ra = 0.25
rc = 1/57
sigmac = 1

# Parametre variables en fonction de l'avion
v = 500;

# Parametres du process
# Temps total
T = 20
# nombre de simulations 
N = 100

"""
Simulation des variables Mat et Mcs qui sont gaussiennes
On fait une simulation multidimentionelle de Mat car on 
connait la matrice de variance convariance
"""

# Nombre de subdivisions.
n = 100;
t = np.linspace(0, T, n);
mean = np.zeros((n,), dtype = float)
cova = np.zeros((n,n), dtype = float)
covc =  np.zeros((n,n), dtype = float)



for i in range(n):
    for j in range(n):
        cova[i,j] = ra**2 * min(t[i],t[j])**2;
        covc[i,j] = sigmac**2 * (1-np.exp(-2*rc*v*min(t[i],t[j])/sigmac))# * np.exp(-rc*v*np.abs(t[i]-t[j])/sigmac)

t2 = time()
print(t2-t1)
# Simulation des vecteurs gaussiens
Ma = np.random.multivariate_normal(mean, cova, size=1)
Mc = np.random.multivariate_normal(mean, covc, size=1)

"""
Simulation du vol d'un avion
"""

Xa = v*t + Ma[0]
Xc = Mc
sig = npl.cholesky(covc)