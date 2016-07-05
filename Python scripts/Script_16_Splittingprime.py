import numpy as np
import numpy.random as npr
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


# Quantile function
def quant(X, alpha):
    G = np.sort(X)
    size = G.size
    index = int(size * alpha)
    return G[index]

def phi(X):
    out = np.min(X, axis = 1)
    return out


def Splitting(distance,Nsim,npoint):
    alpha = 0.5 # quantile level for adapt threshld
    mean = moyenne(npoint,distance)
    cov = covariance(npoint)
    X = np.random.multivariate_normal(mean, cov, size=Nsim)
    rho = 0.5 # param markovian kernel
    nu = np.sqrt(1-rho**2)
    
    S = 0.1 # threshold to be exeeded (epsilon)
    q_alpha =  quant(phi(X), alpha) # Estimation of quantile
    
    eye = np.eye(npoint) # auxiliary
    
    i = 0
    while(q_alpha > S):
        w = (phi(X)<q_alpha) # weights for resampling
        while(np.sum(w)==0):
            X = np.random.multivariate_normal(mean, cov, size=Nsim)
            w = (phi(X)<q_alpha)
        w = w /np.sum(w)
        ind = npr.choice(np.arange(Nsim), size = Nsim, replace = True, p = w) # resampling
        Y = X[ind] # resampling
        p = rho*Y+nu*np.random.multivariate_normal(mean, eye, size=Nsim) # Markovian kernel application
        aux1 = (p.T*(phi(p)<q_alpha)).T
        aux2 = (Y.T*(phi(p)>=q_alpha)).T
        X = aux1 + aux2 # new population
        q_alpha = quant(phi(X), alpha) # position of the next threshold
        i=i+1
        
    proba = (1-alpha)**i * np.mean(phi(X)<S) # probability estimation with splitting
    return proba

npoint = 20
epsilon = 0.1


for distance in np.linspace(4, 8, 3):
    for Nsim in [100, 1000, 100000]:
        A = []
        text_file = open("OutFiles/Output_Splitting_linear_%s_%s.csv" % (distance,Nsim), "w")
        text_file.write("Distance entre avions : %s \n" % distance)
        text_file.write("Nombre de simulations : %s \n" % Nsim)
        text_file.write("Distance, Probability, Error, Relative error, mu \n \n")
        
        for i in range(23):
            prob = Splitting(distance, Nsim, npoint)
            A.append(prob)
            text_file.write("%s, %.3E, %s \n" % (distance, decimal.Decimal(prob), Nsim))
        
        low = epsilon * np.ones(npoint)
        upp = 100 * np.ones(npoint)
        
        mean = moyenne(npoint, distance)
        cov = covariance(npoint)
        p,i = mvn.mvnun(low,upp,mean,cov) # p prob toutes > 0.1
        real = 1-p # prob qu'il existe une < 0.1
        text_file.write("Real value : %s \n" % real)
        estim = np.mean(A)
        text_file.write("%s, %.3E, %.3E, %.3f, %s \n" % (distance, estim, decimal.Decimal(estim-real), 100*(estim-real)/real, Nsim))
        del A[:]
        text_file.close()
        