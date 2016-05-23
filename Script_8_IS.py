# -*- coding: utf-8 -*-
"""
Created on Tue May 17 20:01:37 2016

@author: felipe
"""

import numpy as np
import numpy.linalg as npl
from scipy.linalg import toeplitz

distance = 4.0 # distance(nmi)
Nsim = 10**5 # number of simulations 
npoint = 20.0 # numper of points in the trajectory
Time = 20.0
v=500.0/60.0
ra=0.25
rc=1.0/57
sigmac=1.0



def func(U, dist):
    ind = np.any(U < -dist + 0.1, axis = 1)
    return ind


# Creation of matriecs Ma and Mc
t = np.linspace(0, Time, npoint)
o = np.outer(np.ones(npoint), t)
ot = o.T
minij = np.where(o < ot, o, ot)
    
M1 = 1 - np.exp(-2 * (rc/sigmac) * v * minij)
M2 = sigmac**2 * np.exp(-(rc/sigmac) * v * toeplitz(t,t))

covc = M1 * M2

U = np.random.multivariate_normal(np.zeros(npoint), 2*covc, size=Nsim)


# Monte Carlo method to calculate the probability
ind_mc = func(U, distance)
p_emp_MC = np.mean(ind_mc)
erreur_MC = 1.96*np.sqrt(p_emp_MC*(1-p_emp_MC)/Nsim) 
print("MC estimation")
print(p_emp_MC)
print("MC error")
print(erreur_MC)


##Importance sampling
for dec in  np.linspace(0,4, 100):
    
    delta = dec * np.ones(npoint)
    
    inv = npl.pinv(2 * covc)
    A = U-distance
    
    factor = -np.dot(np.dot(A, inv), delta) - np.dot(np.dot(delta.T, inv), delta)/2 
    
    ech_IS = func(U + dec, distance) * np.exp(factor)
    
    p_emp_IS = np.mean(ech_IS)
    
    var_emp_IS = np.mean(ech_IS**2)
    
    erreur_emp_IS = 1.96*np.sqrt(var_emp_IS - p_emp_IS**2)/np.sqrt(Nsim)
    
    print("IS estimation")
    print(p_emp_IS)
    print(erreur_emp_IS)