# -*- coding: utf-8 -*-
"""
Created on Tue May 17 20:01:37 2016

@author: felipe
"""

import numpy as np
import numpy.linalg as npl


def func(U, epsilon):
    # Computes the prob any (U_i) is less than epsilon
    ind = np.any(U < epsilon, axis = 1)
    return ind

distance = 2.0 # distance(nmi)
print("Distance entre avions")
print(distance)

epsilon = 0.1 # Choc distance
Nsim = 10**5 # number of Monte Carlo simulations 
npoint = 100 # numper of points in the trajectory
Time = 20.0

v=500.0/60.0 # airplane speed
rc=1.0/57 # param
sigmac=1.0 # param


t = np.linspace(3, Time, npoint);
mean = distance * np.ones((npoint,), dtype = float)
cov =  np.zeros((npoint,npoint), dtype = float)


for i in range(npoint):
    for j in range(npoint):
        cov[i,j] = 2 * sigmac**2 * (1-np.exp(-2*rc*v*min(t[i],t[j])/sigmac)) * np.exp(-rc*v*np.abs(t[i]-t[j])/sigmac)

# Simulation des vecteurs gaussiens
Diff = np.random.multivariate_normal(mean, cov, size=Nsim)


# Monte Carlo method to calculate the probability
ind_mc = func(Diff, epsilon)
p_emp_MC = np.mean(ind_mc)
erreur_MC = 1.96*np.sqrt(p_emp_MC*(1-p_emp_MC)/Nsim) 
print("MC estimation")
print(p_emp_MC)
print("MC error")
print(erreur_MC)
print("MC intervalle de confiance")
print([p_emp_MC - erreur_MC, p_emp_MC + erreur_MC])


##Importance sampling
inv = npl.pinv(cov)
aux = Diff - mean
X = []
Y = []
Si = np.linspace(-0, -distance, 20)
for dec in Si:
    delta = dec * np.ones(npoint)
    
    L = -np.dot(np.dot(aux, inv), delta) - np.dot(np.dot(delta.T, inv), delta)/2
    
    ech_IS = func(Diff + delta, epsilon) * np.exp(L)

    p_emp_IS = np.mean(ech_IS)
    
    var_emp_IS = np.var(ech_IS)
    
    erreur_emp_IS = 1.96*np.sqrt(var_emp_IS - p_emp_IS**2)/np.sqrt(Nsim)
    X.append(p_emp_IS)
    Y.append(erreur_emp_IS)

A = np.asarray(Y) # erreur != 0
B = np.asarray(X) # prob id err !=0

i = np.argmin(A[A>0])
p_emp_IS = B[A>0][i]
erreur_emp_IS = A[A>0][i]
print("mu pris")
print(Si[A>0][i])
print("IS estimation")
print(p_emp_IS)
print("IS error")
print(erreur_emp_IS)
print("IS intervalle de confiance")
print([p_emp_IS - erreur_emp_IS, p_emp_IS + erreur_emp_IS])