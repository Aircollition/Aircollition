#Air Collision
Project in Python to simulate Aircraft collision

**Keywords**: Aircraft collision, Stochastic Simulation, Ornstein-Uhlenbeck process, Monte Carlo estimation, Variance reduction methods.

## Introduction
The number of airplanes currently circulating has became considerable. Just by now one aircraft takes off every second, in total there's an average of 80 000 flights by day and in 2020 it is estimated to increase up to 200 millions of flights per year. Therefore preventing the aircraft collision is an important matter and it could be treated in different ways from flight plan designing to protocols to evade immediate collision. In this project we estimate the probability of collision in a two aircraft simulation

## Aircraft path simulation
The airplane path is principally influenced by the wind, this brings an aleatory factor in the flight model. The trajectories are modeled by an Ornstein-Uhlenbeck process according to [4]. More on the model can be found in the report (in french) and in the presentation (in french too). The goal is to estimate the probability of collision of two planes, for that the trajectories are simulated and using variance-reduction methods the probability of collision is calculated.

##How to install the project
To test any of the python snippets just do:

    python file.py
for each of the pdf files just do:

    make pdf


## References
[1] Jacquemart, Damien, and Jérôme Morio. *Conflict probability estimation between aircraft with dynamic importance splitting*. Safety science 51.1 (2013): 94-100.

[2] Morio, Jérôme, and Mathieu Balesdent. *Estimation of Rare Event Probabilities in Complex Aerospace and Other Systems: A Practical Approach*. Woodhead Publishing, 2015.

[3] Paielli, Russell A., and Heinz Erzberger. *Conflict probability for free flight*. Journal of Guidance, Control, and Dynamics 20.3 (1997): 588-596.

[4] Prandini, Maria, and Oliver J. Watkins. *Probabilistic aircraft conflict detection*. HYBRIDGE, IST-2001 32460 (2005): 116-119.

##Licence
MIT Licence, see the [`LICENCE.md`](https://github.com/Aircollition/Aircollition/blob/master/LICENSE.md) file.