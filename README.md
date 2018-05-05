# CriticalForagingOrgs
A community of Ising-embodied organisms adapt via evolution (GA) and/or a 'critical learning' algorithm. The community is contained within an arena with a finite (default 100) number of food parcels which infinitely replenish. The genetic algorithm adapts the community to the task of foraging for food whereas the critical learning algorithm seeks to organize the the ising-embodied organisms towards their critical point (a process independent of the foraging task). These algorithms can be applied seperately or together which can be controlled in the settings defined in the file `train`.

## Running a Simulation
To run your own simulation, simply run:
```
python3 train
```

However, bear in mind that the settings in the `train` python script will need to be edited to your own preferences and directories.

## Ising Class
The Ising class is defined in the `embodied_ising.py` file. This file was originally forked from the project:

https://github.com/MiguelAguilera/Criticality-as-It-Could-Be

and its associated arXiv link:

https://arxiv.org/abs/1704.05255

This file has been heavily modified, retro-fitted, and mutated to generalize the simulations done in the "Criticality as it Could Be" project. Originally this project was looking at learning criticality in a single agent playing a simple game. We generalize the techniques used in the project onto a community of agents. This generalization is done to contextualize criticality to biological systems subject to evolutionary dynamics.

## Critical Learning
The details of this process are explained in the arXiv link posted above.
One of the many requirements for this algorithm is the usage of a distrubution of correlation values sampled from a known critical system. This file is provided `correlations-ising-generalized-size83.npy'.


## Genetic Algorithm (GA)
The genotypes of an individual ising-embodied organism is defined by the connectivity matrix of its neural network (and potentially it's local Beta (inverse temperature) if that setting is turned on). Starting with randomly generated neural networks for each organism, the community is allowed to evolve for some large number of timepoints until the GA is applied. A combination of **elitism** methods to duplicate (with mutations) the top organisms that have consumed the most food as well as **crossover** mating interactions.

There are a number of parameters here that can be modified in the settings in the `train` file which control important details about the GA, for example:
- mutation rates
- sparsity of hidden-neuron interconnectivity
- temperature mutation toggles

## Research Goals
This project is motivated by the growing interest and observation of criticality in nature, in particular criticality in complex, living systems. The relationship between the self-organization of criticality and the dynamics of evolution are still not deeply understood. It seems more and more clear that self-organized criticality and evolution are concepts that overlap in many dimensions and understanding this relationship seems essential to understanding the emergence of complexity in our universe. To this end, this project seeks to simulate an en environment with encompasses a community of agents that can evolve and learn to be critical in the context of the well-studied Ising model. 
