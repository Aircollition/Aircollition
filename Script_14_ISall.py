# -*- coding: utf-8 -*-
"""
Created on Tue May 17 20:01:37 2016

@author: felipe
"""

import numpy as np
import numpy.linalg as npl
import AeroProcess as ap
import matplotlib.pyplot as plt
from scipy.stats import mvn
from LaTeXPy import latexify


Nsim = 10**5
npoint = 20
counter = 0

distance = 4
mean = ap.moyenne(npoint, distance)
cov = ap.covariance1(npoint)
x = np.linspace(0,distance,100)

R = []
P = []

for epsilon in x:
    low = epsilon * np.ones(npoint)
    upp = 100 * np.ones(npoint)
    p,i = mvn.mvnun(low,upp,mean,cov) # p prob toutes > epsilon
    R.append((1-p))

R = np.asarray(R)
P = R[1:] - R[:-1]

latexify()
plt.figure()
plt.bar(x[1:], P, width = 4/99)
plt.title("Density of collition")
plt.xlabel("$\epsilon$")
plt.ylabel("Probability")
plt.savefig('Outputs/Script_14_1.pdf', bbox_inches='tight')


## IS
A =[]
B =[]
C =[]
x = np.linspace(0,4, 20)

for epsilon in x:
    inv = npl.pinv(cov)
    Diff = np.random.multivariate_normal(mean, cov, size=Nsim)
    aux = Diff - mean
    dec = epsilon-4
    a = dec * np.linspace(0,1,npoint/2)
    
    b = dec * np.linspace(1,0,npoint/2)
    #delta = dec * np.linspace(0,1,npoint)
    delta = np.concatenate((a,b))
        
    L = -np.dot(np.dot(aux, inv), delta) - np.dot(np.dot(delta.T, inv), delta)/2
        
    ech_IS = ap.func(Diff + delta, epsilon) * np.exp(L)
    
    p_emp_IS = np.mean(ech_IS)
        
    var_emp_IS = np.var(ech_IS)
        
    erreur_emp_IS = 1.96*np.sqrt(var_emp_IS - p_emp_IS**2)/np.sqrt(Nsim)
    A.append(p_emp_IS)
    B.append(var_emp_IS)
    C.append(erreur_emp_IS)
    
A = np.asarray(A)
X = A[1:] - A[:-1]

plt.figure()
plt.bar(x[1:], np.abs(X), width = 4/19)
plt.title("Density of collition with IS")
plt.xlabel("$\epsilon$")
plt.ylabel("Probability")
plt.savefig('Outputs/Script_14_2.pdf', bbox_inches='tight')


plt.show()
