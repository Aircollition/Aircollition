import numpy as np
import scipy.stats as sp
from scipy.stats import multivariate_normal
from scipy.stats import mvn

dim = 20
samples = 10**5

G = multivariate_normal.rvs(np.zeros(dim), np.eye(dim), size = samples)

# Calcul de la probabilité G > a
# G c'est une gaussienne centre reduite de dim = dim i.e. dim normales indep
# Calcul de la prob qu'il existe une plus grand que a

a = 6

ind_mc = np.any(G > a, axis = 1)
p_emp_MC = np.mean(ind_mc)
erreur_MC = 1.96*np.sqrt(p_emp_MC*(1-p_emp_MC)/samples) 
print("MC estimation")
print(p_emp_MC)
print("MC error")
print(erreur_MC)
print("MC intervalle de confiance")
print([p_emp_MC - erreur_MC, p_emp_MC + erreur_MC])

print("Real value")
real = 1 - sp.norm.cdf(a)**dim
print(real)

## Par IS

dec = 2 # meme decalage pour toutes
#delta = dec * np.ones(dim)
delta = np.zeros(dim)
delta[0] = dec
#delta = dec * np.linspace(0,1,dim)

L = -np.dot(G, delta) - np.dot(delta.T, delta)/2 # likelyhood

ech_IS = np.any(G + delta > a, axis = 1) * np.exp(L)

p_emp_IS = np.mean(ech_IS)

var_emp_IS = np.var(ech_IS)

erreur_emp_IS = 1.96*np.sqrt(var_emp_IS - p_emp_IS**2)/np.sqrt(samples)


print("IS estimation")
print(p_emp_IS)
print("IS error")
print(erreur_emp_IS)
print("IS intervalle de confiance")
print([p_emp_IS - erreur_emp_IS, p_emp_IS + erreur_emp_IS])

low = -10 * np.ones(dim)
upp = a * np.ones(dim)
p,i = mvn.mvnun(low,upp,np.zeros(dim),np.eye(dim))
print(1-p)