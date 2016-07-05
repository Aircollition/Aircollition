import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from scipy.stats import mvn
from LaTeXPy import latexify


def func(U, epsilon):
    # Computes the prob any (U_i) is less than epsilon
    ind = np.any(U < epsilon, axis = 1)
    return ind

def quant(X, alpha):
    G = np.sort(X)
    size = G.size
    index = int(size * alpha)
    return G[index]

def phi(X):
    out = np.min(X, axis = 1)
    return out
    
epsilon = 0.1 # Choc distance
Nsim = 10**5 # number of Monte Carlo simulations 
npoint = 20 # numper of points in the trajectory
Time = 100.0
dmax = 8

v=500.0/60.0 # airplane speed
rc=1.0/57 # param
sigmac=1.0 # param


t = np.linspace(0, Time, npoint);

cov =  np.zeros((npoint,npoint), dtype = float)

for i in range(npoint):
    for j in range(npoint):
        cov[i,j] = 2 * sigmac**2 * (1-np.exp(-2*rc*v*min(t[i],t[j])/sigmac)) * np.exp(-rc*v*np.abs(t[i]-t[j])/sigmac)

A = []
B = []

for distance in np.linspace(0, dmax, 20):
    mean = distance * np.ones((npoint,), dtype = float)
    # FIN DEFINITION DU PROCESSUS
    
    # Simulation des vecteurs gaussiens
    X = np.random.multivariate_normal(mean, cov, size=Nsim)
    
    # Monte Carlo method to calculate the probability
    ind_mc = func(X, epsilon)
    p_emp_MC = np.mean(ind_mc)
    erreur_MC = 1.96*np.sqrt(p_emp_MC*(1-p_emp_MC)/Nsim) 
    
    # Splitting Method
    # Quantile function
    
    N = 10**5
    alpha = 0.5 # quantile level for adapt threshld
    X = np.random.multivariate_normal(mean, cov, size=N)
    rho = 0.5 # param markovian kernel
    nu = np.sqrt(1-rho**2)
    S = 0.1 # threshold to be exeeded
    q_alpha =  quant(phi(X), alpha) # Estimation of quantile
    
    eye = np.eye(npoint) # auxiliary
    
    i = 0
    while(q_alpha > S):
        w = (phi(X)<q_alpha) # weights for resampling
        while(np.sum(w)==0):
            X = np.random.multivariate_normal(mean, cov, size=N)
            w = (phi(X)<q_alpha)
        w = w /np.sum(w)
        ind = npr.choice(np.arange(N), size = N, replace = True, p = w) # resampling
        Y = X[ind] # resampling
        p = rho*Y+nu*np.random.multivariate_normal(mean, eye, size=N) # Markovian kernel application
        aux1 = (p.T*(phi(p)<q_alpha)).T
        aux2 = (Y.T*(phi(p)>=q_alpha)).T
        X = aux1 + aux2 # new population
        q_alpha = quant(phi(X), alpha) # position of the next threshold
        i=i+1
        
    proba = (1-alpha)**i * np.mean(phi(X)<S) # probability estimation with splitting
    A.append(p_emp_MC)
    B.append(proba)

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
plt.semilogy(np.linspace(0, dmax, 20), A, 'rx', label = 'MC')
plt.semilogy(np.linspace(0, dmax, 20), B, 'b.', label ='Splitting')
plt.semilogy(np.linspace(0, 8, 100), P, 'k', label ='num')
plt.xlabel("Separation distance")
plt.ylabel("Probability")
plt.legend()
plt.savefig('Outputs/Script_10_SplittingvsMC.pdf', bbox_inches='tight')