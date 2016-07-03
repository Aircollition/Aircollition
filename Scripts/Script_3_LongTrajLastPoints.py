import AeroProcess as ap
import matplotlib.pyplot as plt

"""
Distribution of the last point in the trajectory
"""
Nsim = 1000
npoint = 100
theta = 0
v = 500/60
Time = 20

Xa, Xc = ap.TrajOblique([0,0], theta, Nsim, npoint, Time=20)

# Plot of the Distribution of last points in space
plt.figure()
for i in range(Nsim):
    plt.plot(Xa[i,-1], Xc[i,-1], 'k.')
plt.grid(True)
plt.title("Lasts points disribution")
plt.xlabel("nmi")
plt.ylabel("nmi")
plt.plot(Xa[0,0], Xc[0,0], 'ro')
plt.plot(v * Time, 0, 'ro')
plt.savefig('Outputs/Script_3_1.pdf', bbox_inches='tight')

# Plot of the last points
plt.figure()
for i in range(Nsim):
    plt.plot(Xa[i,-1], Xc[i,-1], 'k.')
plt.grid(True)
plt.title("Lasts points disribution near the waypoint")
plt.xlabel("nmi")
plt.ylabel("nmi")
plt.plot(v * Time, 0, 'ro')
plt.savefig('Outputs/Script_3_2.pdf', bbox_inches='tight')

plt.show()