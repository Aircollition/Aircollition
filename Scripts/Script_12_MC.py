
import numpy as np
from scipy.linalg import toeplitz
from scipy.stats import mvn
import decimal



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


def MC(distance, Nsim, npoint):
    epsilon = 0.1 # Choc distance
    Time = 20.0
    
    v=500.0/60.0 # airplane speed
    rc=1.0/57 # param
    sigmac=1.0 # param
    
    t = np.linspace(0, Time, npoint)
    o = np.outer(np.ones(npoint), t)
    ot = o.T
    minij = np.where(o < ot, o, ot)
    
    M1 = 1 - np.exp(-2 * (rc/sigmac) * v * minij)
    M2 = sigmac**2 * np.exp(-(rc/sigmac) * v * toeplitz(t,t))
        
    cov = 2 * M1 * M2
    mean = distance * np.ones((npoint,), dtype = float)
    
    # Simulation des vecteurs gaussiens
    Diff = np.random.multivariate_normal(mean, cov, size=Nsim)
    
    # Monte Carlo method to calculate the probability
    ind_mc = func(Diff, epsilon)
    p_emp_MC = np.mean(ind_mc)
    erreur_MC = 1.96*np.sqrt(p_emp_MC*(1-p_emp_MC)/Nsim) 
    return (p_emp_MC, erreur_MC)

npoint = 20
counter = 0

for distance in np.linspace(1, 8, 8):
    counter = counter+1
    text_file = open("OutFiles/Output_MC_%s.txt" % counter, "w")
    text_file.write("Distance entre avions : %s \n" % distance)
    text_file.write("Probability MC, Error, Rel. Error, N \n")
    
    for Nsim in [100, 1000, 100000]:
        for i in range(20):
            prob, err = MC(distance, Nsim, npoint)
            text_file.write("%s, %.2E, %s, %s \n" % (prob, decimal.Decimal(err), 100*err/prob, Nsim))
    
    low = 0.1 * np.ones(npoint)
    upp = 100 * np.ones(npoint)
    
    mean = moyenne(npoint, distance)
    cov = covariance(npoint)
    
    p,i = mvn.mvnun(low,upp,mean,cov) # p prob toutes > 0.1
    real = 1-p # prob qu'il existe une < 0.1
    text_file.write("Real value : %s" % real)
    text_file.close()