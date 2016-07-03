# -*- coding: utf-8 -*-
"""
Created on Tue May 17 20:01:37 2016

@author: felipe
"""

import numpy as np
import numpy.linalg as npl
from scipy.stats import mvn
import matplotlib.pyplot as plt


def func(U, epsilon):
    # Computes the prob any (U_i) is less than epsilon
    ind = np.any(U < epsilon, axis = 1)
    return ind

distance =4.0 # distance(nmi)
print("Distance entre avions")
print(distance)

epsilon = 0.1 # Choc distance
Nsim = 10**5 # number of Monte Carlo simulations 
npoint = 100 # numper of points in the trajectory
Time = 20.0

v=500.0/60.0 # airplane speed
rc=1.0/57 # param
sigmac=1.0 # param


t = np.linspace(0.1, Time, npoint);
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
Z = []
Si = np.linspace(-0, -distance, 20)

## Look for the best decentrage (in terms of error)
for dec in Si:
    a = dec * np.linspace(0,1,npoint/2)
    b = dec * np.linspace(1,0,npoint/2)
    delta = np.concatenate((a,b))
    
    #delta = dec * np.linspace(0,1,npoint)
    
    L = -np.dot(np.dot(aux, inv), delta) - np.dot(np.dot(delta.T, inv), delta)/2
    
    ech_IS = func(Diff + delta, epsilon) * np.exp(L)

    p_emp_IS = np.mean(ech_IS)
    
    var_emp_IS = np.var(ech_IS)
    
    erreur_emp_IS = 1.96*np.sqrt(var_emp_IS - p_emp_IS**2)/np.sqrt(Nsim)
    X.append(p_emp_IS)
    Y.append(erreur_emp_IS)
    Z.append(var_emp_IS)

A = np.asarray(Y) # err != 0
B = np.asarray(X) # prob id err !=0
C = np.asarray(Z) # var

if (np.all(A==0)):
    print("Importance Sampling Estimation fail")
    print("No useful data to compute the probability ")
    print("Try using more simulations (Nsim)")
else:
    P = A[A>0] # array of errors
    Q = B[A>0] # array of probabilities
    R = C[A>0] # array of variances
    
    i = np.argmin(P) # Critère du choix
    
    p_emp_IS = Q[i]
    var_emp_IS = R[i]
    erreur_emp_IS = P[i]
    
    print("mu pris")
    mu = Si[A>0][i]
    print(mu)
    print("IS estimation")
    print(p_emp_IS)
    print("IS error")
    print(erreur_emp_IS)
    print("IS intervalle de confiance")
    print([p_emp_IS - erreur_emp_IS, p_emp_IS + erreur_emp_IS])

low = epsilon * np.ones(npoint)
upp = 100 * np.ones(npoint)
p,j = mvn.mvnun(low,upp,mean,cov)
print("Real value : ")
print(1-p)
print("Percentual error : ")
print(np.abs(p_emp_IS-(1-p))/(1-p) * 100)
#for max in np.linspace(30,100, 10):
#    upp = max * np.ones(npoint)
#    p,i = mvn.mvnun(low,upp,mean,cov)
#    plt.semilogy(max,1-p, 'b.')
#    plt.title("Probability")
#    plt.xlabel("max integration")
#    plt.ylabel("Probability")

plt.figure()
plt.semilogy(Si[A>0], P, 'b', label ='IS')
plt.semilogy(mu, erreur_emp_IS, 'ro')
plt.xlabel("$\mu$ décalage")
plt.ylabel("Erreur")
plt.title("Error at distance %s" % distance)
plt.legend()
plt.savefig('Outputs/Script_8_IS_1.pdf', bbox_inches='tight')

plt.figure()
plt.semilogy(Si[A>0], Q, 'b', label ='IS')
plt.semilogy(Si[A>0], (1-p)*np.ones(Si[A>0].size), 'k', label ='num')
plt.semilogy(mu, p_emp_IS, 'ro')
plt.xlabel("$\mu$ décalage")
plt.ylabel("Probability")
plt.title("Probability with IS at distance %s" % distance)
plt.legend()
plt.savefig('Outputs/Script_8_IS_2.pdf', bbox_inches='tight')
plt.show()
