# -*- coding: utf-8 -*-
"""
Created on Tue May 17 20:01:37 2016

@author: felipe
"""

import numpy as np
import scipy.linalg as sp
from scipy.stats import mvn
from scipy.stats import multivariate_normal


def func(U, epsilon, distance):
    # Computes the prob any (U_i) is less than epsilon
    ind = np.any(U < epsilon - distance, axis = 1)
    return ind

distance =10.0 # distance(nmi)
print("Distance entre avions")
print(distance)

epsilon = 0.1 # Choc distance
Nsim = 10**5 # number of Monte Carlo simulations 
npoint = 20 # numper of points in the trajectory
Time = 20.0

v=500.0/60.0 # airplane speed
rc=1.0/57 # param
sigmac=1.0 # param


t = np.linspace(0.1, Time, npoint);
mean = np.zeros((npoint,), dtype = float)
cov =  np.zeros((npoint,npoint), dtype = float)


for i in range(npoint):
    for j in range(npoint):
        cov[i,j] = 2 * sigmac**2 * (1-np.exp(-2*rc*v*min(t[i],t[j])/sigmac)) * np.exp(-rc*v*np.abs(t[i]-t[j])/sigmac)

# Simulation des vecteurs gaussiens
U = np.random.multivariate_normal(mean, cov, size=Nsim)


# Monte Carlo method to calculate the probability
ind_mc = func(U, epsilon, distance)
p_emp_MC = np.mean(ind_mc)
erreur_MC = 1.96*np.sqrt(p_emp_MC*(1-p_emp_MC)/Nsim) 
print("MC estimation")
print(p_emp_MC)
print("MC error")
print(erreur_MC)
print("MC intervalle de confiance")
print([p_emp_MC - erreur_MC, p_emp_MC + erreur_MC])


##Importance sampling
C = sp.sqrtm(cov)
G = multivariate_normal.rvs(np.zeros(npoint), np.eye(npoint), size = Nsim)
X = []
Y = []
Si = np.linspace(-0, -distance, 20)

## Look for the best decentrage (in terms of error)
for dec in Si:
    dec = -4
    #a = dec * np.linspace(0,1,npoint/2)
    #b = dec * np.linspace(1,0,npoint/2)
    delta = dec * np.linspace(0,1,npoint)
    #delta = np.concatenate((a,b))
    
    L = -np.dot(G, delta) - np.dot(delta.T, delta)/2 # likelyhood
    
    ech_IS = func(np.dot(G + delta,C), epsilon, distance) * np.exp(L)

    p_emp_IS = np.mean(ech_IS)
    
    var_emp_IS = np.var(ech_IS)
    
    erreur_emp_IS = 1.96*np.sqrt(var_emp_IS - p_emp_IS**2)/np.sqrt(Nsim)
    X.append(p_emp_IS)
    Y.append(erreur_emp_IS)

A = np.asarray(Y) # erreur != 0
B = np.asarray(X) # prob id err !=0

if (np.all(A==0)):
    print("Importance Sampling Estimation fail")
    print("No useful data to compute the probability ")
    print("Try using more simulations (Nsim)")
else:
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

low = epsilon * np.ones(npoint)
upp = 100 * np.ones(npoint)
mean = distance * np.ones(npoint)
p,i = mvn.mvnun(low,upp,mean,cov)
print("Real value : ")
print(1-p)