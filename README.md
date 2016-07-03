#Air Collision (in progress...)

Project in Python to simulate Aircraft collision

**Keywords**: Aircraft collision, Stochastic Simulation, Ornstein-Uhlenbeck process, Monte Carlo estimation, Variance reduction methods.

## Introduction
The number of airplanes currently circulating has became considerable. Just by now one aircraft takes off every second, in total there's an average of 80 000 flights by day and in 2020 it is estimated to increase up to 200 millions of flights per year. Therefore preventing the aircraft collision is an important matter and it could be treated in different ways from flight plan designing to protocols to evade immediate collision. In this project we estimate the probability of collision in a two aircraft simulation

## Aircraft path simulation
The airplane path is principally influenced by the wind, this brings an aleatory factor in the flight model. The trajectories are modeled by an Ornstein-Uhlenbeck process according to [4]. the definition is the following: 

![eq1](http://mathurl.com/jjjn23u.png)

where $W_{a,t}$ and $W_{c,t}$ are normalized Brownian motions. To simulate the trajectories, the paths are discretized and the processus is simulated according to the Variance-covariance matrix (when $s <t$)

![eq2](http://mathurl.com/zszj8xo.png)

	The probability that we want to find is the following:
	
![eq3](http://mathurl.com/jorpmr3.png)

with ![eq4](http://mathurl.com/zwmhcpl.png) a predefined minimal distance of collition, for example it could be taken as the total length of two airplane wings.
##How to install the project
Just execute the following lines of code:

    make project
    ipython graphs

## References
[1] Jacquemart, Damien, and Jérôme Morio. *Conflict probability estimation between aircraft with dynamic importance splitting*. Safety science 51.1 (2013): 94-100.

[2] Morio, Jérôme, and Mathieu Balesdent. *Estimation of Rare Event Probabilities in Complex Aerospace and Other Systems: A Practical Approach*. Woodhead Publishing, 2015.

[3] Paielli, Russell A., and Heinz Erzberger. *Conflict probability for free flight*. Journal of Guidance, Control, and Dynamics 20.3 (1997): 588-596.

[4] Prandini, Maria, and Oliver J. Watkins. *Probabilistic aircraft conflict detection*. HYBRIDGE, IST-2001 32460 (2005): 116-119.