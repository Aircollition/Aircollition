import numpy as np
import numpy.random as npr


def func(U, epsilon):
    # Computes the prob any (U_i) is less than epsilon
    ind = np.any(U < epsilon, axis = 1)
    return ind

epsilon = 0.1 # Choc distance
Nsim = 10**5 # number of Monte Carlo simulations 
npoint = 20 # numper of points in the trajectory
Time = 100.0
distance = 7
print("Distance:")
print(distance)

v=500.0/60.0 # airplane speed
rc=1.0/57 # param
sigmac=1.0 # param


t = np.linspace(0, Time, npoint);

mean = distance * np.ones((npoint,), dtype = float)
cov =  np.zeros((npoint,npoint), dtype = float)

for i in range(npoint):
    for j in range(npoint):
        cov[i,j] = 2 * sigmac**2 * (1-np.exp(-2*rc*v*min(t[i],t[j])/sigmac)) * np.exp(-rc*v*np.abs(t[i]-t[j])/sigmac)

# FIN DEFINITION DU PROCESSUS

# Simulation des vecteurs gaussiens
X = np.random.multivariate_normal(mean, cov, size=Nsim)

# Monte Carlo method to calculate the probability
ind_mc = func(X, epsilon)
p_emp_MC = np.mean(ind_mc)
erreur_MC = 1.96*np.sqrt(p_emp_MC*(1-p_emp_MC)/Nsim) 
print("MC estimation")
print(p_emp_MC)
print("MC error")
print(erreur_MC)
print("MC intervalle de confiance")
print([p_emp_MC - erreur_MC, p_emp_MC + erreur_MC])

# Splitting Method
print("Begin splitting Method")

# Quantile function
def quant(X, alpha):
    G = np.sort(X)
    size = G.size
    index = int(size * alpha)
    return G[index]

def phi(X):
    out = np.min(X, axis = 1)
    return out

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

print("Probability estimation with Splitting")
print(proba)
print("Probability with Monte Carlo")
print(p_emp_MC)
print("rel error %")
rel_err = (proba- p_emp_MC)/proba
rel_err = np.abs(rel_err * 100)
print(rel_err)