# -*- coding: utf-8 -*-
"""
Created on Tue May 17 20:01:37 2016

@author: felipe
"""

import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
from scipy.stats import mvn
from LaTeXPy import latexify


def func(U, epsilon):
    # Computes the prob any (U_i) is less than epsilon
    ind = np.any(U < epsilon, axis = 1)
    return ind

epsilon = 0.1 # Choc distance
Nsim = 10**5 # number of Monte Carlo simulations 
npoint = 20 # numper of points in the trajectory
Time = 20.0

v=500.0/60.0 # airplane speed
rc=1.0/57 # param
sigmac=1.0 # param


t = np.linspace(3, Time, npoint);
cov =  np.zeros((npoint,npoint), dtype = float)

for i in range(npoint):
    for j in range(npoint):
        cov[i,j] = 2 * sigmac**2 * (1-np.exp(-2*rc*v*min(t[i],t[j])/sigmac)) * np.exp(-rc*v*np.abs(t[i]-t[j])/sigmac)

inv = npl.pinv(cov)
## MC vs IS
A = []
B = []
C = []
D = []
E = []
F = []
for distance in np.linspace(0, 8, 20):
    # Simulation des vecteurs gaussiens
    mean = distance * np.ones((npoint,), dtype = float)
    Diff = np.random.multivariate_normal(mean, cov, size=Nsim)
    # Monte Carlo method to calculate the probability
    ind_mc = func(Diff, epsilon)
    p_emp_MC = np.mean(ind_mc)
    var_MC = np.var(ind_mc)
    erreur_MC = 1.96*np.sqrt(p_emp_MC*(1-p_emp_MC)/Nsim) 
    A.append(p_emp_MC)
    B.append(erreur_MC)
    E.append(var_MC)
    # IS
    aux = Diff - mean
    X = []
    Y = []
    Z = []
    Si = np.linspace(-0, -distance, 20)
    for dec in Si:
        #delta = dec * np.ones(npoint)
        delta = dec * np.linspace(0,1,npoint)
        
        L = -np.dot(np.dot(aux, inv), delta) - np.dot(np.dot(delta.T, inv), delta)/2
        
        ech_IS = func(Diff + delta, epsilon) * np.exp(L)
        
        p_emp_IS = np.mean(ech_IS)
        
        var_emp_IS = np.var(ech_IS)
        
        erreur_emp_IS = 1.96*np.sqrt(var_emp_IS - p_emp_IS**2)/np.sqrt(Nsim)
        
        X.append(p_emp_IS)
        Y.append(erreur_emp_IS)
        Z.append(var_emp_IS)
    
    M = np.asarray(X)
    N = np.asarray(Y)
    P = np.asarray(Z)
    
    i = np.argmin(N)
    p_emp_IS = M[i]
    erreur_emp_IS = N[i]
    var_emp_IS = P[i]
    
    if(np.any(N>0)):
        i = np.argmin(N[N>0])
        p_emp_IS = M[N>0][i]
        erreur_emp_IS = N[N>0][i]
        var_emp_IS = P[N>0][i]
    
    C.append(p_emp_IS)
    D.append(erreur_emp_IS)
    F.append(var_emp_IS)

low = epsilon * np.ones(npoint)
upp = 100 * np.ones(npoint)
P = []
for distance in np.linspace(0,8,100):
    mean = distance * np.ones(npoint)
    p,i = mvn.mvnun(low,upp,mean,cov)
    P.append(1-p)

latexify()
plt.figure()
plt.grid(True)
plt.semilogy(np.linspace(0, 8, 20), A, 'rx', label = 'MC')
plt.semilogy(np.linspace(0, 8, 20), C, 'b.', label ='IS')
plt.semilogy(np.linspace(0, 8, 100), P, 'k', label ='num')
plt.xlabel("Separation distance")
plt.ylabel("Probability")
plt.legend()
plt.savefig('Outputs/Script_8_ISmc_1.pdf', bbox_inches='tight')

plt.figure()
plt.grid(True)
plt.semilogy(np.linspace(0, 8, 20), B, 'rx', label = 'MC')
plt.semilogy(np.linspace(0, 8, 20), D, 'b.', label = 'IS')
plt.xlabel("Separation distance")
plt.ylabel("Estimation Error (95\%)")
plt.legend()
plt.savefig('Outputs/Script_8_ISmc_2.pdf', bbox_inches='tight')

plt.figure()
plt.grid(True)
plt.semilogy(np.linspace(0, 10, 20), E, 'rx', label = 'MC')
plt.semilogy(np.linspace(0, 10, 20), F, 'b.', label = 'IS')
plt.xlabel("Separation distance")
plt.ylabel("Variance")
plt.legend()

plt.show()