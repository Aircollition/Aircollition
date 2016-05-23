import AeroProcess as ap
import matplotlib.pyplot as plt
import numpy as np

"""
Plot of a two aircrafts paths, Montecarlo simulation 
of collition in function of the distance
"""

distance = 4 # distance(nmi)
Nsim = 10**5 # number of Monte Carlo simulations 
npoint = 100 # numper of points in the trajectory
Time = 20

A1x, A1y = ap.AircraftTraj(Nsim, npoint, Time = Time)
A2x, A2y = ap.AircraftTraj(Nsim, npoint, Time = Time)
A2y += distance


plt.figure()
plt.plot(A1x[0], A1y[0], 'b', lw = 2)
plt.plot(A2x[0], A2y[0], 'r', lw = 2)

plt.grid(True)
plt.xlim([-20, 180])
plt.ylim([-4, distance + 4])
plt.xlabel("nmi")
plt.ylabel("nmi")
plt.plot(A1x[0,0], A1y[0,0], 'ko')
plt.plot(A1x[0,-1], A1y[0,-1], 'ko')
plt.plot(A2x[0,0], A2y[0,0], 'ko')
plt.plot(A2x[0,-1], A2y[0,-1], 'ko')
plt.savefig('Outputs/Script_5_1.pdf', bbox_inches='tight')

# Montecarlo method

currdist = (A1x - A2x) **2 + (A1y - A2y) ** 2
mindist = np.min(currdist, axis = 1)
mindist = np.sqrt(mindist)
timemin = np.argmin(currdist, axis = 1)

# Plot of the separation distance through time of one trajectory
plt.figure()
plt.plot(np.linspace(0,Time, npoint), np.sqrt(currdist[0]))
plt.title("Sep distance through time")
plt.xlabel("time (min)")
plt.ylabel("distance (nmi)")
plt.savefig('Outputs/Script_5_2.pdf', bbox_inches='tight')

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
plt.savefig('Outputs/Script_5_3.pdf', bbox_inches='tight')


# Hist plot of the mindistance for Nsim simulations
plt.figure()
Long = np.max(mindist) - np.min(mindist)
Nbins = int(round(Nsim**(1./3.)*Long/3.49))
plt.hist(mindist, bins = Nbins, range=(0, distance))
plt.title("Density of min sep distance")
plt.xlabel("min sep distance")
plt.ylabel("Number of times")
plt.savefig('Outputs/Script_5_4.pdf', bbox_inches='tight')


# Plot of the distance through time for Nsim simulations
plt.figure()
for i in range(10**3):
    plt.plot(np.linspace(0,Time, npoint), np.sqrt(currdist[i]))
plt.title("Sep distance through time for several samples")
plt.xlabel("time (min)")
plt.ylabel("distance (nmi)")
plt.savefig('Outputs/Script_5_5.pdf', bbox_inches='tight')


plt.show()