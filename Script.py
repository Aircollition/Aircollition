# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 22:41:51 2016

@author: felipe
"""

import Outils as fl
import numpy as np

# Exemples trajectoires

fl.PlotFlight(5)
fl.PlotFlight(100)
fl.PlotFlight(1000)

# Exemple de plot des trajectoires obliques
fl.PlotTraj(5, 10**3, T=0.1)

# Exemple de calcul de la probabilité 
# de collision (distance < 0.1 nmi) pour des trajectoires parallèles
# Par la methode de montecarlo pour des distances croissantes
# Pour des distances entre 0 et 5
nbSimulations = 1000
Time = 60

print("Methode MC pour la probabilite de collision ")
for dist in np.linspace(0,5, 10):
    prob = fl.MontecarloParallele(dist,nbSimulations,T=Time)
    s = 'distance : ' + str(dist)[0:4] + ' - probabilite : ' + str(prob)
    print(s)

# Le resultat est petite pour distances longues
# et ça precision depend fortement de le nombre de simulations

fl.PlotCroisse()

# Exemple de calcul de la probabilité 
# de collision (distance < 0.1 nmi) pour des trajectoires en parallèles
# Par la methode de montecarlo pour des distances croissantes
# Pour des vitesses croissantes entre 0 et 5
nbSimulations = 10000
Time = 60
print("Methode MC pour la probabilite de collision a des vitesses variables")
for vit in np.linspace(500/3 - 10,500/3 + 10, 10):
    prob = fl.MontecarloParallele(1, nbSimulations, T=Time, v1 = 500/3, v2 = vit)
    s = 'vitesse : ' + str(vit)[0:6] + ' - probabilite : ' + str(prob)
    print(s)
    
# Exemple de calcul d'impact pour des trajets en croix
print("Methode MC pour la probabilite de collision traj en croix")
for i in np.arange(10):
    prob = fl.MontecarloCroix(nbSimulations,T=20)
    s = 'probabilite : ' + str(prob)
    print(s)