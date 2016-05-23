import AeroProcess as ap
import matplotlib.pyplot as plt
import numpy as np

"""
Plot of a trajectory of 30 degrees between two waypoints
at speed 500/60 nmi/min and 
"""
Nsim = 20
npoint = 100
theta = 45
v = 500/60
Time = 20

Xa, Xc = ap.TrajOblique([0,0], theta, Nsim, npoint, Time=Time, v =v)

plt.figure()
for i in range(Nsim):
    plt.plot(Xa[i], Xc[i], 'b')

plt.grid(True)
plt.xlabel("nmi")
plt.ylabel("nmi")
plt.plot(Xa[0,0], Xc[0,0], 'ro')
plt.plot(v * Time * np.cos(np.radians(theta)), v * Time * np.sin(np.radians(theta)), 'ro')
plt.savefig('Outputs/Script_2_1.pdf', bbox_inches='tight')
plt.show()