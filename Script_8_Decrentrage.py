# -*- coding: utf-8 -*-
"""
Calculation of collition via decentrage

@author: felipe
"""

import AeroProcess as ap
import numpy as np
import numpy.linalg as npl
from scipy.linalg import toeplitz

distance = 4 # distance(nmi)
Nsim = 10**5 # number of simulations 
npoint = 200 # numper of points in the trajectory
Time = 20
v=500/60
ra=0.25
rc=1/57
sigmac=1



def func(U):
    ind = np.any(np.abs(U) < 0.1, axis = 1)
    return ind

A1x, A1y = ap.AircraftTraj(Nsim, npoint, Time = Time)
A2x, A2y = ap.AircraftTraj(Nsim, npoint, Time = Time)
A2y+= distance

# Creation of matriecs Ma and Mc
t = np.linspace(1, Time, npoint)
o = np.outer(np.ones(npoint), t)
ot = o.T
minij = np.where(o < ot, o, ot)
    
M1 = 1 - np.exp(-2 * (rc/sigmac) * v * minij)
M2 = sigmac**2 * np.exp(-(rc/sigmac) * v * toeplitz(t,t))

covc = M1 * M2

# Monte Carlo method to calculate the probability
U = A2y - A1y
ind_mc = func(U)
p_emp_MC = np.mean(ind_mc)
erreur_MC = 1.96*np.sqrt(p_emp_MC*(1-p_emp_MC)/Nsim) 
print("MC estimation")
print(p_emp_MC)
print("MC error")
print(erreur_MC)


##Importance sampling
dec = - distance  # decalage


delta = dec * np.ones(npoint)
inv = npl.inv(2 * covc)

A = U-distance

ech_IS = func(U + dec) * np.exp(-np.dot(np.dot(A, inv), delta) - np.dot(np.dot(delta.T, inv), delta)/2)

p_emp_IS = np.mean(ech_IS)

var_emp_IS = np.mean(ech_IS**2)

erreur_emp_IS = 1.96*np.sqrt(var_emp_IS - p_emp_IS**2)/np.sqrt(Nsim)

print("IS estimation")
print(p_emp_IS)
print(erreur_emp_IS)
#p_theorique = 1-norm.cdf(a)
