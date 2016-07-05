import numpy as np
import matplotlib.pyplot as plt

npoint = 100
dec = 3

x = np.linspace(0,1,npoint)

a = dec * np.linspace(0,1,npoint/2)
b = dec * np.linspace(1,0,npoint/2)
delta = np.concatenate((a,b))

delta1 = dec * np.linspace(0,1,npoint)

delta2 = dec * np.ones(npoint)

plt.figure()
plt.plot(x, delta2)

plt.figure()
plt.plot(x, delta1)

plt.figure()
plt.plot(x, delta)
