# -*- coding: utf-8 -*-
"""
Created on Tue May 17 20:01:37 2016

@author: felipe
"""

import numpy as np
import numpy.linalg as npl
from scipy.stats import mvn


def func(U, epsilon):
    # Computes the prob any (U_i) is less than epsilon
    ind = np.any(U < epsilon, axis = 1)
    return ind
    
def covariance(npoint):
    v=500.0/60.0 # airplane speed
    rc=1.0/57 # param
    sigmac=1.0 # param
    Time = 20.0
    t = np.linspace(0.1, Time, npoint);
    cov =  np.zeros((npoint,npoint), dtype = float)
    for i in range(npoint):
        for j in range(npoint):
            cov[i,j] = 2 * sigmac**2 * (1-np.exp(-2*rc*v*min(t[i],t[j])/sigmac)) * np.exp(-rc*v*np.abs(t[i]-t[j])/sigmac)
    return cov

def moyenne(npoint, distance):
    mean = distance * np.ones((npoint,), dtype = float)
    return mean

def IS(distance, Nsim, npoint):
    
    epsilon = 0.1 # Choc distance
    
    mean = moyenne(npoint,distance)
    cov =  covariance(npoint)
    # Simulation des vecteurs gaussiens
    Diff = np.random.multivariate_normal(mean, cov, size=Nsim)
    
    
    ##Importance sampling
    inv = npl.pinv(cov)
    aux = Diff - mean
    X = []
    Y = []
    Si = np.linspace(-0, -distance, 20)
    
    ## Look for the best decentrage (in terms of error)
    for dec in Si:
        
        # Uncomment to do toit
        #a = dec * np.linspace(0,1,npoint/2)
        #b = dec * np.linspace(1,0,npoint/2)
        #delta = np.concatenate((a,b))
        
        # Uncomment to do linear
        #delta = dec * np.linspace(0,1,npoint)
        
        # Uncomment to do equal
        delta = dec * np.ones(npoint)
        
        # Uncomment change only one value
        #delta = dec * np.zeros(npoint)
        #delta[int(npoint/2)] = dec
        
        L = -np.dot(np.dot(aux, inv), delta) - np.dot(np.dot(delta.T, inv), delta)/2
        
        ech_IS = func(Diff + delta, epsilon) * np.exp(L)
    
        p_emp_IS = np.mean(ech_IS)
        
        var_emp_IS = np.var(ech_IS)
        
        erreur_emp_IS = 1.96*np.sqrt(var_emp_IS - p_emp_IS**2)/np.sqrt(Nsim)
        X.append(p_emp_IS)
        Y.append(erreur_emp_IS)
    
    A = np.asarray(Y) # erreur != 0
    B = np.asarray(X) # prob id err !=0
    
    if(np.any(A>0)):
        i = np.argmin(A[A>0])
        p_emp_IS = B[A>0][i]
        erreur_emp_IS = A[A>0][i]
        return (p_emp_IS, erreur_emp_IS, Si[A>0][i])
    
    return (0,0,0)


Nsim = 10**5
npoint = 20
counter = 0

for distance in np.linspace(1, 10, 10):
    counter = counter+1
    text_file = open("OutFiles/Output_IS_constant_%s.txt" % counter, "w")
    text_file.write("Distance entre avions : %s \n" % distance)
    text_file.write("Nombre de simulations : %s \n" % Nsim)
    
    for i in range(23):
        prob, err, mu = IS(distance, Nsim, npoint)
        text_file.write("Prob IS: %s ; Error : %s ; mu : %s \n" % (prob, err, mu))
    
    low = 0.1 * np.ones(npoint)
    upp = 100 * np.ones(npoint)
    
    mean = moyenne(npoint, distance)
    cov = covariance(npoint)
    p,i = mvn.mvnun(low,upp,mean,cov) # p prob toutes > 0.1
    real = 1-p # prob qu'il existe une < 0.1
    text_file.write("Real value : %s" % real)
    text_file.close()