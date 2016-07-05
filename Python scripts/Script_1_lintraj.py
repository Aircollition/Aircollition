import AeroProcess as ap
import matplotlib.pyplot as plt

"""
Plot of a linear trajectory between two waypoints
at speed 500/60 nmi/min and 
"""
Nsim = 3
npoint = 100

Xa, Xc = ap.AircraftTraj(Nsim, npoint, Time = 20)

plt.figure()
plt.plot(Xa[0], Xc[0], lw = 2)

plt.grid(True)
plt.xlim([-20, 180])
plt.ylim([-4, 4])
plt.xlabel("nmi")
plt.ylabel("nmi")
plt.text(-0.5,0.5,'start')
plt.text(160,-1,'end')
plt.plot(Xa[0,0], Xc[0,0], 'ko')
plt.plot(Xa[0,-1], Xc[0,-1], 'ko')
plt.savefig('Outputs/Script_1_1.pdf', bbox_inches='tight')
plt.show()