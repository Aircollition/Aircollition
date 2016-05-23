# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 13:39:56 2016

@author: felipe garcia
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
"""
Definitions de quelques fonctions utiles pour la simulation de
trajectoires d'avions.
"""

def Process(n, N, T=20, v=500/3, ra=0.25, rc=1/57, sigmac=1):
    """
    Retourne N trajectoires d'avion en forme de tuple Xa, Xc pour 
    la position along-track et cross-track.
    Simulation de N trajectoires d'avion par une discretisation
    du processus Gaussienn en n variables aleatoires avec matrice de
    variance-covariance :
    (Pour t < s)
    Ka(t,s) = ra**2 * t**2;
    Kc(t,s) = sigmac**2 * (1-exp(-2*rc*v*t/sigmac)) * exp(-rc * v * (s-t)/siagmac) 
    
    Facteurs:
    n : Nombre de points sur la trajectoire
    N : Nombre de simulations
    T : Temps de vol (min)
    v : vitesse d'un avion (standard 500 kt = 500/3 nmi/min)
    Facteurs "fixes"
    ra : facteur du process (default 0.25)
    rc : facteur du process (default 1/57)
    sigmac : facteur du process (default 1)
    """
    t = np.linspace(0, T, n)
    o = np.outer(np.ones(n), t)
    ot = o.T
    cova = ra**2 * np.where(o < ot, o, ot)**2    
    o = np.outer(np.ones(n), t)
    ot = o.T
    
    M1 = 1 - np.exp(-2 * (rc/sigmac) * v * np.where(o < ot, o, ot))
    M2 = np.exp(-(rc/sigmac) * v * toeplitz(t,t))
    
    covc = sigmac**2 * M1 * M2
    mean = np.zeros((n,))
    Ma = np.random.multivariate_normal(mean, cova, size=N)
    Mc = np.random.multivariate_normal(mean, covc, size=N)
    Xa = v*t + Ma
    Xc = Mc
    return (Xa, Xc)

def Sim(Xi, Xf, n, N, v=500/3):
    """
    Retourne N trajectoires d'avion en forme de tuple X, Y
    qui vont du point initial Xi jusqu'à Xf.
    Xi : point initial
    Xf : point final
    n : nombre de points sur chaque trajectoire
    N : nombre de simulations
    """
    Xi = np.asarray(Xi)
    Xf = np.asarray(Xf)
    diff = Xf-Xi
    dist = np.sqrt(np.sum(diff**2))
    time = dist/v
    # On simule un processus de la meme longueur
    Xa, Xc = Process(n, N, T=time, v=v)
    # On fait une rotation de ce processus
    # pour avoir la fin en Xf
    c, s = diff[0]/dist, diff[1]/dist
    Xar, Xcr =Xi[0] + c*Xa-s*Xc, Xi[1] + s*Xa+c*Xc
    return (Xar, Xcr)

def MontecarloParallele(dist, N, n = 10**3, T=20, v1=500/3, v2 = 500/3):
    """
    Simulation monte carlo de deux trajectoires parallèles
    avec une distance dist entre elles. Retourne le probabilité
    de collition.
    Paramètres:
    N : nombre de simulations
    n: nombre de points sur trajectoire (1000 c'est assez bien)
    v1 : vitesse du premièr avion
    v2 : vitesse du deuxième avion
    T : temps de passage sur waypoint
    """
    # Simulation des trajectoires
    Xa1, Xc1 = Sim([0, 0], [0,T*v1], n, N)
    Xa2, Xc2 = Sim([dist, 0], [dist, T*v2], n, N)
    
    # Calcul de la probabilité
    airdist = np.sqrt((Xa2-Xa1)**2 + (Xc2-Xc1)**2)
    mindist = np.min(airdist, axis = 1)
    
    return np.sum(mindist < 0.1)/N
    
def WhereConflic(dist, N, n = 10**3, T=20, v1=500/3, v2 = 500/3):

    # Simulation des trajectoires
    Xa1, Xc1 = Sim([0, 0], [0,T*v1], n, N)
    Xa2, Xc2 = Sim([dist, 0], [dist, T*v2], n, N)
    
    # Calcul du temps
    airdist = np.sqrt((Xa2-Xa1)**2 + (Xc2-Xc1)**2)
    mindist = np.argmin(airdist, axis = 1)
   
    return mindist * (airdist[mindist] < 0.1) * T/n
    
def MontecarloCroix(N, n = 10**3, T=20, v1=500/3, v2 = 500/3):
    """
    Simulation monte carlo de deux trajectoires en croix
    avec une distance dist entre elles. Retourne le probabilité
    de collition.
    Paramètres:
    N : nombre de simulations
    n: nombre de points sur trajectoire (1000 c'est assez bien)
    v1 : vitesse du premièr avion
    v2 : vitesse du deuxième avion
    T : temps de passage sur waypoint
    """
    # Simulation des trajectoires
    theta = np.sqrt(2)/2
    
    Xa1, Xc1 = Sim([0, 0], [T * v1 * theta, T * v1 * theta], n, N)
    Xa2, Xc2 = Sim([T * v2 * theta, 0], [0, T * v2 * theta], n, N)
    
    # Calcul de la probabilité
    airdist = np.sqrt((Xa2-Xa1)**2 + (Xc2-Xc1)**2)
    mindist = np.min(airdist, axis = 1)
    
    return np.sum(mindist < 0.1)/N

def PlotFlight(dist,n = 100, T=20, v1=500/3, v2 = 500/3):
    """
    Plot d'une trajectoire de deux avions en parallèle    
    """
    Xa, Xc = Sim([0, 0], [0,3 * T*v1], n, 1)
    Xar, Xcr = Sim([dist, 3 * T*v2], [dist, 0], n, 1)
    plt.figure()
    plt.plot(Xa[0], Xc[0], 'r')
    plt.plot(Xar[0], Xcr[0], 'b')
    
    plt.grid(True)
    plt.xlabel("nmi")
    plt.ylabel("nmi")
    plt.text(0,0,'Avion 1')
    plt.text(dist,0,'Avion 2')
    
    # Waypoints
    Waypointsx = [0, 0, 0, 0, dist, dist, dist, dist]
    Waypointsy = [0, T*v1, 2*T*v1, 3*T*v1, 0, T*v2, 2*T*v2, 3*T*v2]
    plt.plot(Waypointsx, Waypointsy, 'ko')
    plt.show()
    return 
    
def PlotCroix(n = 10**3, T=20, v1=500/3, v2 = 500/3):
    
    theta = np.sqrt(2)/2
    Xa1, Xc1 = Sim([0, 0], [T*v1 * theta, T*v1 * theta], n, 1)
    Xa2, Xc2 = Sim([T*v2 * theta, 0], [0, T*v2 * theta], n, 1)
    
    plt.figure()
    plt.plot(Xa1[0], Xc1[0], 'r')
    plt.plot(Xa2[0], Xc2[0], 'b')
    
    plt.grid(True)
    plt.xlabel("nmi")
    plt.ylabel("nmi")
    plt.text(0,0,'Avion 1')
    plt.text(T*v2 * theta, 0,'Avion 2')
    return
    
def PlotTraj(N, n, T=20, v=500/3):
    alpha = np.sqrt(2)/2
    Xa, Xc = Sim([0, 0], [T*v * alpha, T*v * alpha], n, N)
    
    plt.figure()
    for i in np.arange(N):
        plt.plot(Xa[i], Xc[i])
        
    plt.grid(True)
    plt.xlabel("nmi")
    plt.ylabel("nmi")
    plt.text(0,0,'Avion 1')
    plt.show()
    return
    
def PlotCroisse(n = 10**3, T=20, v=500/3):
    """
    Simulation des trajectoires croissés
    """
    d = T*v
    r3 = np.sqrt(3)/2
    
    Xa, Xc = Sim([0, 0], [0, d], n, 1)
    Xa1, Xc1 = Sim([0, d], [0 + d*r3, d + d/2], n, 1)
    Xa2, Xc2 = Sim([0 + d*r3, d + d/2], [0 + 2*d*r3, d + 2*d/2], n, 1)
    X1 = np.concatenate((Xa,Xa1,Xa2), axis=1)
    Y1 = np.concatenate((Xc,Xc1,Xc2), axis=1)
    
    Xar, Xcr = Sim([0, d + 2*d/2], [d, d + 2*d/2], n, 1)
    Xa1r, Xc1r = Sim([d, d + 2*d/2], [d + d/2, d + 2*d/2 - d*r3], n, 1)
    Xa2r, Xc2r = Sim([d + d/2, d + 2*d/2 - d*r3], [d + 2*d/2, d + 2*d/2 - 2*d*r3], n, 1)
    X2 = np.concatenate((Xar,Xa1r,Xa2r), axis=1)
    Y2 = np.concatenate((Xcr,Xc1r,Xc2r), axis=1)
    plt.figure()
    plt.plot(X1[0], Y1[0], 'r')
    plt.plot(X2[0], Y2[0], 'b')

    plt.grid(True)
    plt.text(0,0,'Avion 1')
    plt.text(0, d + 2*d/2,'Avion 2')
    
    Waypointsx = [0, 0, d*r3, 2*d*r3, 0, d, d+d/2, 2*d]
    Waypointsy = [0, d, d + d/2, d + 2*d/2, d + 2*d/2, d + 2*d/2, d + 2*d/2 - d*r3, d + 2*d/2 - 2*d*r3]
    plt.plot(Waypointsx, Waypointsy, 'ko')
    
    plt.show()
    return
    