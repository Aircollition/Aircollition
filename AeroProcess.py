"""
@author: Felipe Garcia

Methods for simulating an Aircraft trajectory
in accordance wind and flight parameters 
"""
import numpy as np
from scipy.linalg import toeplitz

def AircraftTraj(Nsim, npoint, Time=20, v=500/60, ra=0.25, rc=1/57, sigmac=1):
    """
    Outputs N Aircraft trajectories in form of tuple Xa, Xc
    from point (0,0) horizontally
    la position along-track et cross-track.
    Simulation de N trajectoires d'avion par une discretisation
    du processus Gaussienn en n variables aleatoires avec matrice de
    variance-covariance :
    (Pour t < s)
    Ka(t,s) = ra**2 * t**2;
    Kc(t,s) = sigmac**2 * (1-exp(-2*rc*v*t/sigmac)) * exp(-rc * v * (s-t)/siagmac) 
    
    Facteurs:
    n : Nombre de points sur la trajectoire
    N : Nombre de simulations
    T : Temps de vol (min)
    v : vitesse d'un avion (standard 500 kt = 500/60 nmi/min)
    Facteurs "fixes"
    ra : facteur du process (default 0.25)
    rc : facteur du process (default 1/57)
    sigmac : facteur du process (default 1)
    """
    t = np.linspace(0, Time, npoint)
    o = np.outer(np.ones(npoint), t)
    ot = o.T
    minij = np.where(o < ot, o, ot)
    cova = ra ** 2 * minij ** 2    
    
    M1 = 1 - np.exp(-2 * (rc/sigmac) * v * minij)
    M2 = sigmac**2 * np.exp(-(rc/sigmac) * v * toeplitz(t,t))
    
    covc = M1 * M2
    
    Xa = np.random.multivariate_normal(v * t, cova, size=Nsim)
    Xc = np.random.multivariate_normal(np.zeros(npoint), covc, size=Nsim)
    return (Xa, Xc)
    
def TrajOblique(Xi, theta, Nsim, npoint, Time=20, v=500/60, ra=0.25, rc=1/57, sigmac=1):
    """
    Outputs N Aircraft trajectories in form of tuple Xa, Xc
    qui vont du point initial Xi jusqu'à Xf.
    Xi : point initial
    Xf : point final
    n : nombre de points sur chaque trajectoire
    N : nombre de simulations
    """
    Xi = np.asarray(Xi)
    # On simule un processus de la meme longueur
    Xa, Xc = AircraftTraj(Nsim, npoint, Time=Time, v=v, ra=ra, rc=rc, sigmac=sigmac)
    # On fait une rotation de ce processus
    # pour avoir la fin en Xf
    theta = np.radians(theta)
    c, s = np.cos(theta), np.sin(theta)
    
    Xar, Xcr =Xi[0] + c*Xa-s*Xc, Xi[1] + s*Xa+c*Xc
    return (Xar, Xcr)

def Covariance(npoint, Time=20, v=500/60, ra=0.25, rc=1/57, sigmac=1):
    """
    Outputs the covariance matrix cova and covb 
    defined like:
    (Pour t < s)
    Ka(t,s) = ra**2 * t**2;
    Kc(t,s) = sigmac**2 * (1-exp(-2*rc*v*t/sigmac)) * exp(-rc * v * (s-t)/siagmac) 
    """
    t = np.linspace(0, Time, npoint)
    o = np.outer(np.ones(npoint), t)
    ot = o.T
    minij = np.where(o < ot, o, ot)
    cova = ra ** 2 * minij ** 2    
    
    M1 = 1 - np.exp(-2 * (rc/sigmac) * v * minij)
    M2 = sigmac**2 * np.exp(-(rc/sigmac) * v * toeplitz(t,t))
    
    covc = M1 * M2
    
    return(cova, covc)

def func(U, epsilon):
    """
    Computes if any U_i is less than epsilon
    Used to compute the conflict probability
    that a variable (distance between planes)
    is less than 0.1 i.e. collition
    """
    ind = np.any(U < epsilon, axis = 1)
    return ind

def quant(X, alpha):
    """
    function to determine empirical
    quantile functionused for the splitting 
    method
    """
    G = np.sort(X)
    size = G.size
    index = int(size * alpha)
    return G[index]

def phi(X):
    """
    Used as func to compute the probability 
    that a variable (distance between planes)
    is less than 0.1 i.e. collition
    """
    out = np.min(X, axis = 1)
    return out
    
def normalize(w):
    """
    Normalize a weight vector
    to sum 1. Used in splitting 
    method with resampling
    """
    w = w /np.sum(w)
    return w

def mc(X, Nsim, epsilon=0.1):
    """
    Help function that outputs
    the monte carlo empirical probability
    and it's error (at 95%)
    """
    ind_mc = func(X, epsilon)
    p_emp_MC = np.mean(ind_mc)
    erreur_MC = 1.96*np.sqrt(p_emp_MC*(1-p_emp_MC)/Nsim) 
    return (p_emp_MC, erreur_MC)

def resample(X, q_alpha, mean, cov, Nsim):
    w = (phi(X)<q_alpha) # weights for resampling
    
    while(np.sum(w)==0):
        X = np.random.multivariate_normal(mean, cov, size=Nsim)
        w = (phi(X)<q_alpha)
        
    w = normalize(w)
    ind = np.random.choice(np.arange(Nsim), size = Nsim, replace = True, p = w) # resampling
    Y = X[ind] # resampling
    return Y