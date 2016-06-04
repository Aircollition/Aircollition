import AeroProcess as ap
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npl

distance = 4 # distance(nmi)
Nsim = 10**5 # number of Monte Carlo simulations 
npoint = 20 # numper of points in the trajectory
Time = 20

A1x, A1y = ap.AircraftTraj(Nsim, npoint, Time = Time)
A2x, A2y = ap.TrajOblique([20,-50], 30, Nsim, npoint)

plt.figure()
plt.plot(A1x[0], A1y[0], 'b', lw = 2)
plt.plot(A2x[0], A2y[0], 'r', lw = 2)

# Montecarlo method

currdist = (A1x - A2x) **2 + (A1y - A2y) ** 2
currdist = np.sqrt(currdist)
mindist = np.min(currdist, axis = 1)
mindist = np.sqrt(mindist)
timemin = np.argmin(currdist, axis = 1)

# Probability of collition
ind = mindist < 0.1
prob = np.mean(ind)
var = np.var(ind)
erreur = 1.96*np.sqrt(var - prob**2)/np.sqrt(Nsim)
print("MC estimation")
print(prob)
print("Intervalle de conficance")
print([prob-erreur, prob+erreur])

# Plot of the distance through time for Nsim simulations
plt.figure()
for i in range(10**3):
    plt.plot(np.linspace(0,Time, npoint), np.sqrt(currdist[i]))
plt.title("Sep distance through time for several samples")
plt.xlabel("time (min)")
plt.ylabel("distance (nmi)")
plt.savefig('Outputs/Script_15_5.pdf', bbox_inches='tight')


## IS

def f(x,y, delta):
    ind = np.any(x**2+y**2 < delta**2, axis = 1)
    return ind

epsilon = 0.1
cova, covc = ap.covs(npoint)
zero = np.zeros((npoint,npoint))
A = np.concatenate((cova, zero))
B = np.concatenate((zero, covc))
COV = np.concatenate((A,B), axis = 1)

inv = npl.pinv(COV)

a = np.zeros(npoint)
b = 500/60 * np.linspace(0,Time,npoint)
mean = np.concatenate((a,b))
U1 = A1x-A2x
U2 = A1y-A2y
aux = Diff - mean

dec = -2 # TODO choix de delta
a = dec * np.linspace(0,1,npoint/2)
    
b = dec * np.linspace(1,0,npoint/2)
#delta = dec * np.linspace(0,1,npoint)
delta = np.concatenate((a,b))
        
L = -np.dot(np.dot(aux, inv), delta) - np.dot(np.dot(delta.T, inv), delta)/2
        
ech_IS = f(U1, U2, epsilon) * np.exp(L)
    
p_emp_IS = np.mean(ech_IS)
        
var_emp_IS = np.var(ech_IS)
        
erreur_emp_IS = 1.96*np.sqrt(var_emp_IS - p_emp_IS**2)/np.sqrt(Nsim)