import AeroProcess as ap
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npl
from scipy.stats import mvn
from LaTeXPy import latexify

distance = 4 # distance(nmi)
Nsim = 10**5 # number of Monte Carlo simulations 
npoint = 20 # numper of points in the trajectory
Time = 20

A1x, A1y = ap.AircraftTraj(Nsim, npoint, Time = Time)
A2x, A2y = ap.TrajOblique([20,-50], 30, Nsim, npoint)

latexify()
plt.figure()
plt.plot(A1x[0], A1y[0], 'b', lw = 2, label = 'Avion 1')
plt.plot(A2x[0], A2y[0], 'r', lw = 2, label = 'Avion 2')
plt.legend()
plt.grid(True)
plt.xlabel("nmi")
plt.ylabel("nmi")

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
plt.savefig('Outputs/Script_15_1.pdf', bbox_inches='tight')


## IS

def f(x,y, delta):
    ind = np.any(x**2+y**2 < delta**2, axis = 1)
    return ind
    
def covs(npoint, Time=20, v=500/60, ra=0.25, rc=1/57, sigmac=1):

    t = np.linspace(0.1, Time, npoint)
    o = np.outer(np.ones(npoint), t)
    ot = o.T
    minij = np.where(o < ot, o, ot)
    cova = ra ** 2 * minij ** 2    
    
    M1 = 1 - np.exp(-2 * (rc/sigmac) * v * minij)
    M2 = sigmac**2 * np.exp(-(rc/sigmac) * v * toeplitz(t,t))
    
    covc = M1 * M2
    
    return (cova, covc)

epsilon = 0.1
npoint =20
theta = 30
theta = np.deg2rad(theta)

cova, covc = ap.covs(npoint)

zero = np.zeros((npoint,npoint))
A = np.concatenate((cova, zero))
B = np.concatenate((zero, covc))

COV = np.concatenate((A,B), axis = 1)

S1 = np.cos(theta)**2 * cova + np.sin(theta)**2 * covc
Del = np.sin(theta)*np.cos(theta)*(cova-covc)

A = np.concatenate((S1, Del))
B = np.concatenate((Del, S1))

COV1 = np.concatenate((A,B), axis = 1)


a = 500/60 * np.linspace(0,Time,npoint)
b = np.zeros(npoint)
mean = np.concatenate((a,b))

c = a*np.cos(theta) + 20
d = a*np.sin(theta) - 50

mean1 = np.concatenate((c,d))

U1 = A1x-A2x
U2 = A1y-A2y
U = np.concatenate((U1,U2), axis=1)

meanU = mean - mean1
covU = COV + COV1

inv = npl.inv(covU)

aux = U - meanU

for i in range(10**3):
    plt.plot(np.linspace(0,Time, npoint), U1[i])
plt.savefig('Outputs/Script_15_ISx.pdf', bbox_inches='tight')

plt.figure()
for i in range(10**3):
    plt.plot(np.linspace(0,Time, npoint), U2[i])
plt.savefig('Outputs/Script_15_ISy.pdf', bbox_inches='tight')

plt.figure()
for i in range(10**3):
    plt.plot(np.linspace(0,Time, npoint), U1[i]+a)

plt.figure()
for i in range(10**3):
    plt.plot(np.linspace(0,Time, npoint), U2[i]+b)


a = np.linspace(0,0,10)
a1 = np.linspace(10,0,5)
a2 = np.linspace(0,0,5)
a = np.concatenate((a,a1,a2))


b = np.linspace(0,0,10)
b1 = np.linspace(-5,-2,2)
b2 = np.linspace(0,6,4)
b3 = np.linspace(0,0,4)
b = np.concatenate((b,b1,b2,b3))
#d = dec * np.linspace(1,0,npoint/2)
delta = np.concatenate((a,b))

        
L = -np.dot(np.dot(aux, inv), delta) - np.dot(np.dot(delta.T, inv), delta)/2
        
ech_IS = f(U1 + a, U2 + c, epsilon) * np.exp(L)
    
p_emp_IS = np.mean(ech_IS)
        
var_emp_IS = np.var(ech_IS)
        
erreur_emp_IS = 1.96*np.sqrt(var_emp_IS - p_emp_IS**2)/np.sqrt(Nsim)

print("IS estimation")
print(p_emp_IS)
print("IS error")
print(erreur_emp_IS)
print("IS intervalle de confiance")
print([p_emp_IS - erreur_emp_IS, p_emp_IS + erreur_emp_IS])

npoint = 100

m = meanU[50:80]
m1 = meanU[150:180]
m = np.concatenate((m,m1))

c = covU[50:80][150:180]

low = -epsilon * np.ones(2*npoint)
upp = epsilon * np.ones(2*npoint)
p,i = mvn.mvnun(low,upp, meanU,covU)
print("Real value : ")
print(p)
print("Percentual error : ")
print(np.abs(p_emp_IS-p)/p * 100)