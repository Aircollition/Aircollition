import AeroProcess as ap
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import toeplitz
import numpy.linalg as npl
from LaTeXPy import latexify
import decimal


"""
Plot of a two aircrafts paths crossed, Montecarlo simulation 
of collition in function of the distance
"""

def f(x,y, delta):
    ind = np.any(x**2+y**2 < delta**2, axis = 1)
    return ind

tet = 60 # Angle between trajectories
v = 500/60
Time = 20
# d1 and d2 set to zero makes the worst case
d1 = 5 # Distance from midpoint 1 traj
d2 = 5 # Distance from midpoint 1 traj
dist1 = d1 + Time * v / 2 - np.cos(np.radians(tet)) * Time * v / 2 # distance(nmi)
dist2 = d2 + -np.sin(np.radians(tet)) * Time * v / 2 # distance(nmi)
Nsim = 10**5 # number of Monte Carlo simulations 
npoint = 100 # numper of points in the trajectory

A1x, A1y = ap.AircraftTraj(Nsim, npoint, Time = Time)
A2x, A2y = ap.TrajOblique([dist1,dist2], tet, Nsim, npoint, Time=Time)

latexify()
plt.figure()
plt.grid(True)
plt.xlabel("nmi")
plt.ylabel("nmi")
plt.plot(A1x[0], A1y[0], 'b', lw = 2, label = 'avion 1')
plt.plot(A2x[0], A2y[0], 'r', lw = 2, label = 'avion 2')
plt.legend()

plt.grid(True)
plt.xlabel("nmi")
plt.ylabel("nmi")

plt.plot(A1x[0,0], A1y[0,0], 'ko')
plt.plot(A1x[0,-1], A1y[0,-1], 'ko')
plt.plot(A2x[0,0], A2y[0,0], 'ko')
plt.plot(A2x[0,-1], A2y[0,-1], 'ko')
plt.legend()
plt.savefig('Outputs/Script_6_1.pdf', bbox_inches='tight')


# Montecarlo method

currdist = (A1x - A2x) **2 + (A1y - A2y) ** 2
mindist = np.min(currdist, axis = 1)
mindist = np.sqrt(mindist)
timemin = np.argmin(currdist, axis = 1)

# Plot of the separation distance through time of one trajectory
plt.figure()
plt.grid(True)
plt.plot(np.linspace(0,Time, npoint), np.sqrt(currdist[0]))
#plt.title("Sep distance through time")
plt.xlabel("time (min)")
plt.ylabel("distance (nmi)")
plt.savefig('Outputs/Script_6_2.pdf', bbox_inches='tight')


# Plot of the distance through time for Nsim simulations
plt.figure()
plt.grid(True)
for i in range(10**3):
    plt.plot(np.linspace(0,Time, npoint), np.sqrt(currdist[i]))
plt.title("Sep distance through time for several samples")
plt.xlabel("time (min)")
plt.ylabel("distance (nmi)")
plt.savefig('Outputs/Script_6_3.pdf', bbox_inches='tight')


# Probability of collition
prob = np.mean(mindist < 0.1)
print(prob)

ind = mindist < 0.1

# Hist plot of the number of collitions in function of the mindistance
# Conditioned to collition (mindist < 0.1)
plt.figure()
plt.hist(mindist[ind], bins = 4, range=(0, 0.1)) # Distribution of distances less than 0.1
plt.title("Density of min sep distance conditioned to collition")
plt.xlabel("min sep distance")
plt.ylabel("Number of times")
plt.savefig('Outputs/Script_6_4.pdf', bbox_inches='tight')


# Hist plot of the mindistance for Nsim simulations
plt.figure()
Long = np.max(mindist) - np.min(mindist)
Nbins = int(round(Nsim**(1./3.)*Long/3.49))
distance = max(mindist)

plt.hist(mindist, bins = 100, range=(0, distance))
plt.title("Density of min sep distance")
plt.xlabel("min sep distance")
plt.ylabel("Number of times")
plt.savefig('Outputs/Script_6_5.pdf', bbox_inches='tight')


# Zoom

def covs(npoint, t, v=500/60, ra=0.25, rc=1/57, sigmac=1):

    o = np.outer(np.ones(npoint), t)
    ot = o.T
    minij = np.where(o < ot, o, ot)
    cova = ra ** 2 * minij ** 2    
    
    M1 = 1 - np.exp(-2 * (rc/sigmac) * v * minij)
    M2 = sigmac**2 * np.exp(-(rc/sigmac) * v * toeplitz(t,t))
    
    covc = M1 * M2
    
    return (cova, covc)
    
Nsim=10**5
I = [10-3, 10+3]
t = np.linspace(I[0], I[1], npoint)
cova,covc = covs(npoint,t)

Xa = np.random.multivariate_normal(v * t, cova, size=Nsim)
Xc = np.random.multivariate_normal(np.zeros(npoint), covc, size=Nsim)

theta = np.radians(tet)
c, s = np.cos(theta), np.sin(theta)
Ya = np.random.multivariate_normal(v * t, cova, size=Nsim)
Yc = np.random.multivariate_normal(np.zeros(npoint), covc, size=Nsim)
    
Ya, Yc =dist1 + c*Ya-s*Yc, dist2 + s*Ya+c*Yc


plt.figure()
plt.grid(True)
plt.plot(Xa[0], Xc[0], 'b', lw = 2, label = 'avion 1')
plt.plot(Ya[0], Yc[0], 'r', lw = 2, label = 'avion 2')
plt.savefig('Outputs/Script_6_6.pdf', bbox_inches='tight')


U1 = Xa-Ya
U2 = Xc-Yc
ind_mc = f(U1,U2, 0.1)
p_mc = np.mean(ind_mc)
print(p_mc)


# Test covariances well defined
#Nsim=10**5
#T1 = np.random.multivariate_normal(mean, COV, size=Nsim)
#T2 = np.random.multivariate_normal(mean1, COV1, size=Nsim)

#plt.figure()
#plt.plot(T1[0,0:npoint], T1[0,npoint:(2*npoint)], 'b', lw = 2)
#plt.plot(T2[0,0:npoint], T2[0,npoint:(2*npoint)], 'r', lw = 2)

    
def IS(npoint,Nsim,dec, dec1, dist1, dist2, theta):
    t = np.linspace(I[0], I[1], npoint)
    cova,covc = covs(npoint,t)
    
    zero = np.zeros((npoint,npoint))
    A = np.concatenate((cova, zero))
    B = np.concatenate((zero, covc))
    
    COV = np.concatenate((A,B), axis = 1)
    
    S1 = np.cos(theta)**2 * cova + np.sin(theta)**2 * covc
    Del = np.sin(theta)*np.cos(theta)*(cova-covc)
    
    A = np.concatenate((S1, Del))
    B = np.concatenate((Del, S1))
    
    COV1 = np.concatenate((A,B), axis = 1)
    
    
    a = v * t
    b = np.zeros(npoint)
    mean = np.concatenate((a,b))
    
    c = v * t * np.cos(theta) + dist1
    d = v * t * np.sin(theta) + dist2
    mean1 = np.concatenate((c,d))
    
    meanU = mean - mean1
    covU = COV + COV1

    epsilon =0.1
    
    U = np.random.multivariate_normal(meanU, covU, size=Nsim)
    aux = U - meanU
    inv = npl.inv(covU)
    
    U1 = U[:,0:npoint]
    U2 = U[:,npoint:(2*npoint)]
    
    
    a = dec * np.linspace(1,1,npoint)
    b = dec1 * np.linspace(1,1,npoint)
    delta = np.concatenate((a,b))
    
    
    L = -np.dot(np.dot(aux, inv), delta) - np.dot(np.dot(delta.T, inv), delta)/2
            
    ech_IS = f(U1 + a, U2 + b, epsilon) * np.exp(L)
        
    p_emp_IS = np.mean(ech_IS)
            
    var_emp_IS = np.var(ech_IS)
            
    erreur_emp_IS = 1.96*np.sqrt(var_emp_IS - p_emp_IS**2)/np.sqrt(Nsim)
    
    return (p_emp_IS,var_emp_IS,erreur_emp_IS)


theta = np.deg2rad(tet)

plt.show()


npoint = 100
for Nsim in [100, 1000, 100000]:
    X = []
    text_file = open("OutFiles/Output_CrossTraj_%s.csv" % (Nsim), "w")
    text_file.write("Probability, Error, Relative error, mu \n \n")
    
    for i in range(23):
        p,var, err = IS(npoint,Nsim,3, 3, dist1, dist2, theta)
        text_file.write("%s,%s,%s \n" % (p,err,Nsim))
        X.append(p)
        
    text_file.write("\nAccumulated: \n")
    m = np.mean(X)
    err = 1.96*np.sqrt(np.var(X))/np.sqrt(Nsim)
    rel = err/m
    text_file.write("%.3E, %.3E, %.2f\%%, %s \n" % (decimal.Decimal(m), decimal.Decimal(err), rel, Nsim))
    text_file.close()