

import plotting
import numpy as np
import operator
from itertools import combinations, product
# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import copy

from math import atan2
from math import cos
from math import degrees
from math import floor
from math import radians
from random import random
from random import sample
from random import randint
from math import sin
from math import sqrt
from random import uniform
from copy import deepcopy
import multiprocessing as mp
import sys
import os
import pickle
import time
#import random
from tqdm import tqdm
from shutil import copyfile



# ------------------------------------------------------------------------------+
# ------------------------------------------------------------------------------+
# --- CLASSES ------------------------------------------------------------------+
# ------------------------------------------------------------------------------+
# ------------------------------------------------------------------------------+
settings = {}

class ising:
    # Initialize the network
    def __init__(self, settings, netsize, Nsensors=2, Nmotors=2, name=None):  # Create ising model

        self.size = netsize
        self.Ssize = Nsensors  # Number of sensors
        self.Msize = Nmotors  # Number of sensors
        self.radius = settings['org_radius']

        self.h = np.zeros(netsize) # TODO: is this bias, does this ever go over [0 0 0 0 0]???????

        # self.J = np.zeros((self.size, self.size))

        self.J = np.random.random((self.size, self.size))*2 - 1
        self.J = (self.J + self.J.T) / 2 #Connectivity Matrix
        np.fill_diagonal(self.J, 0)

        self.max_weights = 2

        self.maxRange = sqrt((settings['x_max'] - settings['x_min']) ** 2 +
                             (settings['y_max'] - settings['y_min']) ** 2)

        self.randomize_state()
        self.xpos = 0.0 # Position
        self.ypos = 0.0
        self.randomize_position(settings) #randomize position

        # self.r = uniform(0, 360)  # orientation   [0, 360]
        # self.v = uniform(0, settings['v_max']/3)  # velocity      [0, v_max]
        # self.dv = uniform(-settings['dv_max'], settings['dv_max'])  # dv


        self.dx = 0
        self.dy = 0

        self.name = name
        self.generation = 0
        '''
        initial beta
        '''
        self.Beta = settings['init_beta']
        #self.Beta = 1.0
        # self.defaultT = max(100, netsize * 20)

        self.Ssize1 = 1 # FOOD ROTATIONAL SENSOR: sigmoid(theta)
        self.Ssize2 = 1 # FOOD DISTANCE SENSOR: sigmoid(distance)
        self.Ssize3 = 1 # DIRECTIONAL NEIGHBOUR SENSOR: dot-product distance normalized, see self.org_sens

        self.Msize1 = int(self.Msize/2)  # dv motor neuron


        # MASK USED FOR SETTINGS J/h TO 0
        self.maskJ = np.ones((self.size, self.size), dtype=bool)
        self.maskJ[0:self.Ssize, 0:self.Ssize] = False
        self.maskJ[-self.Msize: -self.Msize] = False
        self.maskJ[0:self.Ssize, -self.Msize:] = False
        np.fill_diagonal(self.maskJ, 0)
        self.maskJ = np.triu(self.maskJ)

        self.J[~self.maskJ] = 0

        # self.maskJtriu = np.triu(self.maskJ)

        self.disconnect_hidden_neurons(settings)

        self.maskh = np.ones(self.size, dtype=bool)
        self.maskh[0:self.Ssize] = False

        self.d_food = self.maxRange  # distance to nearest food
        self.r_food = 0  # orientation to nearest food
        self.org_sens = 0 # directional, 1/distance ** 2 weighted organism sensor
        self.fitness = 0
        self.energy = 0.0
        self.food = 0
        self.energies = [] #Allows for using median as well... Replace with adding parameter up for average in future to save memory? This array is deleted before saving to reduce file size
        self.avg_energy = 0
        self.all_velocity = 0
        self.avg_velocity = 0
        self.v = 0.0

        if settings['share_food']:
            self.foodfound = 0
            self.foodshared = 0
            self.foodgiven = 0

        self.assign_critical_values(settings)


        # if not settings['BoidOn']:
        #     self.Update(settings, 0)

    def reset_state(self, settings):

        # randomize internal state (not using self.random_state since it also randomizes sensors)
        self.s = np.random.random(size=self.size) * 2 - 1
        # randomize position (not using self.randomize_position function since it also randomizes velocity)
        self.xpos = uniform(settings['x_min'], settings['x_max'])  # position (x)
        self.ypos = uniform(settings['y_min'], settings['y_max'])  # position (y)

        self.dv = 0
        self.v = 0

        self.ddr = 0
        self.dr = 0

        self.food = 0
        self.fitness = 0

        if settings['energy_model']:
            self.energies = []  # Clear .energies, that .avg_energy is calculated from with each iteration
            self.energy = settings['initial_energy']  # Setting initial energy

            self.avg_energy = 0
            self.all_velocity = 0
            self.avg_velocity = 0

        if settings['share_food']:
            self.foodfound = 0
            self.foodshared = 0
            self.foodgiven = 0

    def get_state(self, mode='all'):
        if mode == 'all':
            return self.s
        elif mode == 'motors':
            return self.s[-self.Msize:]
        elif mode == 'sensors':
            return self.s[0:self.Ssize]
        elif mode == 'non-sensors':
            return self.s[self.Ssize:]
        elif mode == 'hidden':
            return self.s[self.Ssize:-self.Msize]

    def get_state_index(self, mode='all'):
        return bool2int(0.5 * (self.get_state(mode) + 1))

    # Randomize the state of the network
    def randomize_state(self):
        self.s = np.random.randint(0, 2, self.size) * 2 - 1
        self.s = np.array(self.s, dtype=float)

        # SEE SENSOR UPDATE
        # random sensor states are generated by considering the sensor limitations

        random_rfood = (np.random.rand() * 360) - 180
        self.s[0] = random_rfood / 180

        random_dfood = np.random.rand() * self.maxRange
        self.s[1] = np.tanh(self.radius / (random_dfood ** 2 + 1e-6)) * 2 - 1

        random_dorg = np.random.rand() * self.maxRange
        self.s[2] = np.tanh(random_dorg) * 2 - 1

    def randomize_position(self, settings):
        '''
        Only used in TimeEvolve2
        '''
        self.xpos = uniform(settings['x_min'], settings['x_max'])  # position (x)
        self.ypos = uniform(settings['y_min'], settings['y_max'])  # position (y)

        if settings['BoidOn']:
            self.v = (np.random.randn(2) * 2 - 1) * settings['v_max']
            self.dv = (np.random.randn(2) * 2 - 1) * settings['dv_max']
            self.dx = self.v[0] * settings['dt']
            self.dy = self.v[1] * settings['dt']
            # self.r = np.abs(np.arctan(self.ypos / self.xpos))
            self.r = np.arctan2(self.v[1], self.v[0]) * 180 / np.pi
            
            '''
            If 'BoidOn' == False self.r is undefined (it is defined in the upper if condition)
            --> fixed by moving definition of self.r up in else condition
            '''
            
        else:
            self.r = np.random.rand() * 360 
            self.v = np.random.rand() * settings['v_max'] #TODO: This cannot work with huge v_max
            self.dv = np.random.rand() * settings['dv_max']
            self.dx = self.v * cos(radians(self.r)) * settings['dt']
            self.dy = self.v * sin(radians(self.r)) * settings['dt']


    '''
    NOT USED
    # Set random bias to sets of units of the system
    def random_fields(self, max_weights=None):
        if max_weights is None:
            max_weights = self.max_weights
        self.h[self.Ssize:] = max_weights * (np.random.rand(self.size - self.Ssize) * 2 - 1)
    '''
    # Set random connections to sets of units of the system
    def random_wiring(self, max_weights=None):  # Set random values for h and J
        if max_weights is None:
            max_weights = self.max_weights
        for i in range(self.size):
            for j in np.arange(i + 1, self.size):
                if i < j and (i >= self.Ssize or j >= self.Ssize):
                    self.J[i, j] = (np.random.rand(1) * 2 - 1) * self.max_weights

    def updateAcceleration(self):
        self.ddr = (np.sum(self.s[-self.Msize:-self.Msize1]) / 2)
        self.dv = (np.sum(self.s[-self.Msize1:]) / 2)

    def updateVelocity(self, settings):

        self.updateAcceleration()
        self.ddr = self.ddr * settings['ddr_max'] * settings['dt']
        self.dv = self.dv * settings['dv_max'] * settings['dt']

        if settings['energy_model']:
            energy_cost = self.dv * settings['cost_speed']
            if self.energy >= energy_cost and self.dv > settings['dv_min']:
                #if agent has enough energy and wants to accelerate faster than the "free" acceleration
                self.energy -= energy_cost
            elif self.dv > settings['dv_min']:
                #if agent wants to go faster than min speed but does not have energy
                self.dv = settings['dv_min']

        # UPDATE VELOCITY - Motor neuron s.[-self.Msize1:]
        self.v += self.dv - settings['friction'] * self.v**2
        self.dr += self.ddr - settings['friction'] * np.sign(self.dr)*self.dr

        if self.v < 0:
            self.v = 0


        if self.v > settings['v_max']:
            self.v = settings['v_max']

        if settings['energy_model']:
            self.all_velocity += self.v

        if np.abs(self.dr) > settings['dr_max']:
            self.dr = settings['dr_max']


    def Move(self, settings):
        self.updateVelocity(settings)

        # UPDATE POSITION
        self.dx = self.v * cos(radians(self.r)) * settings['dt']
        self.dy = self.v * sin(radians(self.r)) * settings['dt']
        self.xpos += self.dx
        self.ypos += self.dy
        # print(self.dx, self.dy)

        # UPDATE HEADING
        self.r += self.dr
        self.r = self.r % 360

        # periodic boundary conditions.
        self.xpos = (self.xpos + settings['x_max']) % settings['x_max']
        self.ypos = (self.ypos + settings['y_max']) % settings['y_max']

    def MoveOld(self, settings):

        self.r += (np.sum(self.s[-self.Msize:-self.Msize1]) / 2) * settings['dr_max'] * settings['dt']
        self.r = self.r % 360

        self.dv = (np.sum(self.s[-self.Msize1:]) / 2) * settings['dv_max'] * settings['dt']
        self.v += self.dv

        if self.v < 0:
            self.v = 0

        if self.v > settings['v_max']:
            self.v = settings['v_max']

        if settings['energy_model']:
            energy_cost = self.v * settings['cost_speed']
            if self.energy >= energy_cost and self.v > settings['v_min']:
                #if agent has enough energy and wants to accelerate faster than the "free" acceleration
                self.energy -= energy_cost
            elif self.v > settings['v_min']:
                #if agent wants to go faster than min speed but does not have energy
                self.v = settings['v_min']
            self.all_velocity += self.v

        #############################

        # UPDATE POSITION
        self.dx = self.v * cos(radians(self.r)) * settings['dt']
        self.dy = self.v * sin(radians(self.r)) * settings['dt']
        self.xpos += self.dx
        self.ypos += self.dy
        # print(self.dx, self.dy)

        # UPDATE HEADING
        self.r += self.dr
        self.r = self.r % 360

        # periodic boundary conditions.
        self.xpos = (self.xpos + settings['x_max']) % settings['x_max']
        self.ypos = (self.ypos + settings['y_max']) % settings['y_max']

    def MoveVelMotors(self, settings):

        self.r += (np.sum(self.s[-self.Msize:-self.Msize1]) / 2) * settings['dr_max'] * settings['dt']
        self.r = self.r % 360

        self.v = (np.sum(self.s[-self.Msize1:]) / 2) * settings['v_max'] * settings['dt']

        if self.v < 0:
            self.v = 0

        # if self.v > settings['v_max']:
        #     self.v = settings['v_max']

        if settings['energy_model']:
            energy_cost = self.v * settings['cost_speed']
            if self.energy >= energy_cost and self.v > settings['v_min']:
                #if agent has enough energy and wants to accelerate faster than the "free" acceleration
                self.energy -= energy_cost
            elif self.v > settings['v_min']:
                #if agent wants to go faster than min speed but does not have energy
                self.v = settings['v_min']
            self.all_velocity += self.v

        #############################

        # UPDATE POSITION
        self.dx = self.v * cos(radians(self.r)) * settings['dt']
        self.dy = self.v * sin(radians(self.r)) * settings['dt']
        self.xpos += self.dx
        self.ypos += self.dy
        # print(self.dx, self.dy)

        # UPDATE HEADING
        self.r += self.dr
        self.r = self.r % 360

        # periodic boundary conditions.
        self.xpos = (self.xpos + settings['x_max']) % settings['x_max']
        self.ypos = (self.ypos + settings['y_max']) % settings['y_max']

    def UpdateSensors(self, settings):
        # self.s[0] = sigmoid(self.r_food / 180)
        # self.s[1] = sigmoid(self.d_food)

        # normalize these values to be between -1 and 1
        # TODO: make the numberators (gravitational constants part of the connectivity matrix so it can be mutated)
        self.s[0] = self.r_food / 180 # self.r_food can only be -180:180
        # self.s[1] = np.tanh(np.log10(self.radius / (self.d_food ** 2 + 1e-6)))  # self.d_food goes from 0 to ~
        # self.s[2] = np.tanh(np.log10(self.org_sens + 1e-10))
        self.s[1] = np.tanh(self.radius / (self.d_food ** 2 + 1e-6))*2 - 1  # self.d_food goes from 0 to ~
        self.s[2] = np.tanh((self.org_sens))*2 - 1
        # print(self.s[0:3])
    
    # Execute step of the Glauber algorithm to update the state of one unit
    
    def GlauberStep(self, i=None):
        '''
        Utilizes: self.s, self.h, self.J
        Modifies: self.s
        '''
        if i is None:
            i = np.random.randint(self.size)
        eDiff = 2 * self.s[i] * (self.h[i] + np.dot(self.J[i, :] + self.J[:, i], self.s))
        #deltaE = E_f - E_i = -2 E_i = -2 * - SUM{J_ij*s_i*s_j}
        #self.J[i, :] + self.J[:, i] are added because value in one of both halfs of J seperated by the diagonal is zero

        if self.Beta * eDiff < np.log(1.0 / np.random.rand() - 1):
            #transformed  P = 1/(1+e^(deltaE* Beta)
            self.s[i] = -self.s[i]
    '''
    # Execute step of the Glauber algorithm to update the state of one unit
    # Faster version??
    def GlauberStep(self, i=None):
        #if i is None:
        #    i = np.random.randint(self.size) <-- commented out as not used
        eDiff = np.multiply(np.multiply(2, self.s[i]), np.add(self.h[i], np.dot(np.add(self.J[i, :], self.J[:, i]), self.s)))
        if np.multiply(self.Beta, eDiff) < np.log(1.0 / np.random.rand() - 1):  # Glauber
            self.s[i] = -self.s[i]
    '''

    # Execute time-step using an ANN algorithm to update the state of all units
    def ANNStep(self):

        # SIMPLE MLP
        # TODO: add biases (add to GA as well)

        af = lambda x: np.tanh(x)  # activation function
        Jhm = self.J + np.transpose(self.J)  # connectivity for hidden/motor layers


        Jh = Jhm[:, self.Ssize:-self.Msize]  # inputs to hidden neurons
        Jm = Jhm[:, -self.Msize:]  # inputs to motor neurons

        bh = self.h[self.Ssize:-self.Msize]  # biases for hidden neurons
        bm = self.h[-self.Msize:]  # biases for motor neurons

        # activate and update
        new_h = af( self.Beta * ( np.dot(self.s, Jh) + bh ) )
        self.s[self.Ssize:-self.Msize] = new_h

        new_m = af( np.dot(self.s, Jm) + bm )
        self.s[-self.Msize:] = new_m

        #  TODO: non-symmetric Jhm, need to change through to GA



    # Compute energy difference between two states with a flip of spin i
    def deltaE(self, i):
        return 2 * (self.s[i] * self.h[i] + np.sum(
            self.s[i] * (self.J[i, :] * self.s) + self.s[i] * (self.J[:, i] * self.s)))

    # Update states of the agent from its sensors
    def Update(self, settings, i=None):
        if i is None:
            i = np.random.randint(self.size)
        if i == 0:
            self.Move(settings)
            self.UpdateSensors(settings)
        elif i >= self.Ssize:
            self.GlauberStep(i)

    def SequentialUpdate(self, settings):
        for i in np.random.permutation(self.size):
            self.Update(settings, i)


    # Update all states of the system without restricted influences
    def SequentialGlauberStep(self, settings):
        thermalTime = int(settings['thermalTime'])

        self.UpdateSensors(settings) # update sensors at beginning

        # update all other neurons a bunch of times
        for j in range(thermalTime):
            perms = np.random.permutation(range(self.Ssize, self.size))
            #going through all neuron exceot sensors in random permutations
            for i in perms:
                self.GlauberStep(i)

        self.Move(settings) # move organism at end


    # Update all states of the system without restricted influences
    def ANNUpdate(self, settings):
        thermalTime = int(settings['thermalTime'])

        self.UpdateSensors(settings)  # update sensors at beginning

        # update all other neurons a bunch of times
        for j in range(thermalTime):
            self.ANNStep()

        # self.Move(settings)  # move organism at end
        # self.MoveOld(settings)
        self.MoveVelMotors(settings)

    # update everything except sensors
    def NoSensorGlauberStep(self):
        perms = np.random.permutation(range(self.Ssize, self.size))
        for i in perms:
            self.GlauberStep(i)

    # update sensors using glauber steps (dream)
    def DreamSensorGlauberStep(self):
        perms = np.random.permutation(self.size)
        for i in perms:
            self.GlauberStep(i)

    # ensure that not all of the hidden neurons are connected to each other
    def disconnect_hidden_neurons(self, settings):
        numHNeurons = self.size - self.Ssize - self.Msize
        perms = list(combinations(range(self.Ssize, self.Ssize + numHNeurons), 2))
        numDisconnectedEdges = len(list(combinations(range(settings['numDisconnectedNeurons']), 2)))

        for i in range(0, numDisconnectedEdges):
            nrand = np.random.randint(len(perms))
            iIndex = perms[nrand][0]
            jIndex = perms[nrand][1]

            self.J[iIndex,jIndex] = 0
            # self.J[jIndex, iIndex] = 0

            self.maskJ[iIndex, jIndex] = False
            # self.maskJ[jIndex, iIndex] = False

        # self.maskJtriu = np.triu(self.maskJ)

    def assign_critical_values(self, settings):
        # LOAD ISING CORRELATIONS
        # filename = 'correlations-ising2D-size400.npy'
        # Cdist = np.load(filename)
        Cdist = settings['Cdist']

        self.m1 = np.zeros(self.size)
        self.C1 = np.zeros((self.size, self.size))
        for ii in range(self.size):
            for jj in range(max(ii + 1, self.Ssize), self.size):
                ind = np.random.randint(len(Cdist))
                self.C1[ii, jj] = Cdist[ind]

    # re-sort the assigned correlations from the critical ising model so that their order matches the order of the
    # actual correlations
    def sort_critical_correlations(self):
        c = self.C
        x = np.arange(np.prod(c.shape)).reshape(c.shape)[self.maskJ]  # index vector
        c = c[self.maskJ]

        c1 = self.C1[self.maskJ]

        orderc = np.argsort(c)
        orderc1 = np.argsort(c1)

        C1_new = np.zeros((self.size, self.size))

        # loop through index vector and re-sort assigned correlations to match order of actual correlations
        # for iEdge, index in enumerate(x):
        #     i_index = int(np.floor(index / self.size))
        #     j_index = int(index % self.size)
        #
        #     condition = np.subtract(orderc1, orderc[iEdge]) == 0
        #     # C1_index = int(np.extract(condition, orderc1))
        #     # C1_new[i_index, j_index] = c1[C1_index]
        #
        #     C1_new[i_index, j_index] = c1[condition]
        #     # C1_new[i_index, j_index] = c1[orderc1[condition]]

        for i, iEdge in enumerate(orderc):
            index = x[iEdge]
            i_index = int(np.floor(index / self.size))
            j_index = int(index % self.size)

            # condition = np.subtract(orderc1, orderc[i]) == 0

            C1_new[i_index, j_index] = c1[orderc1[i]]

        self.C1 = C1_new

    # mutate the connectivity matrix of an organism by stochastically adding/removing an edge
    def mutate(self, settings):
        '''
        Adds/removes a random edge depending on sparsity setting and randomly mutates another random edge

        :param:  settings
        '''

        # expected number of disconnected edges
        numDisconnectedEdges = len(list(combinations(range(settings['numDisconnectedNeurons']), 2)))
        totalPossibleEdges = len(list(combinations(range(self.size - self.Ssize - self.Msize), 2)))

        # number of (dis)connected edges
        connected = copy.deepcopy(self.maskJ)

        disconnected = ~connected
        np.fill_diagonal(disconnected, 0)
        disconnected = np.triu(disconnected)

        # keep sensors connected to hidden neurons
        # TODO: allow the sensors to disconnect to some hidden neurons.
        #   Make sure minimum of 1 connection is made
        connected[0:self.Ssize, :] = 0
        connected[:, -self.Msize:] = 0
        # things that need to be disconnected and not flagged to change
        disconnected[0:self.Ssize, -self.Msize:] = 0
        disconnected[0:self.Ssize, 0:self.Ssize] = 0

        numEdges = np.sum(connected)
        # positive value means too many edges, negative value means too little
        edgeDiff = numEdges - (totalPossibleEdges - numDisconnectedEdges)
        # edgeDiff = numEdges - numDisconnectedEdges

        # TODO: investigate the empty connectivity matrix here
        prob = sigmoid(edgeDiff)  # probability near 1 means random edge will be removed, near 0 means random edge added
        rand = np.random.rand()

        if prob >= rand:
            # remove random edge
            i, j = np.nonzero(connected)
            if len(i) > 0:
                randindex = np.random.randint(0, len(i))
                ii = i[randindex]
                jj = j[randindex]

                self.maskJ[ii, jj] = False
                self.J[ii, jj] = 0

                # TODO: is this a good way of making the code multi-purpose?
                try:
                    self.C1[ii, jj] = 0
                except NameError:
                    pass

            else:
                print('Connectivity Matrix Empty! Mutation Blocked.')

        else:
            # add random edge
            i, j = np.nonzero(disconnected)
            if len(i) > 0:
                randindex = np.random.randint(0, len(i))
                ii = i[randindex]
                jj = j[randindex]

                self.maskJ[ii, jj] = True
                self.J[ii, jj] = np.random.uniform(-1, 1) * self.max_weights
                # I.J[ii, jj] = np.random.uniform(np.min(I.J[I.Ssize:-I.Msize, I.Ssize:-I.Msize]) / 2,
                #                                 np.max(I.J[I.Ssize:-I.Msize, I.Ssize:-I.Msize]) * 2)
                try:
                    self.C1[ii, jj] = settings['Cdist'][np.random.randint(0, len(settings['Cdist']))]
                except NameError:
                    pass

            else:  # if connectivity matrix is full, just change an already existing edge
                i, j = np.nonzero(connected)

                randindex = np.random.randint(0, len(i))
                ii = i[randindex]
                jj = j[randindex]

                # self.J[ii, jj] = np.random.uniform(-1, 1) * self.max_weights
                # self.J[ii, jj] = np.clip(self.J[ii, jj] * np.random.normal(),
                #                          -self.max_weights, self.max_weights)
                self.J[ii, jj] = np.clip(self.J[ii, jj] + np.random.normal(loc=0, scale=settings['mutationSigma']),
                                         -self.max_weights, self.max_weights)

        # MUTATE RANDOM EDGE
        i, j = np.nonzero(self.maskJ)

        randindex = np.random.randint(0, len(i))
        ii = i[randindex]
        jj = j[randindex]

        # self.J[ii, jj] = np.random.uniform(-1, 1) * self.max_weights
        # self.J[ii, jj] = np.clip(self.J[ii, jj] * np.random.normal(),
        #                          -self.max_weights, self.max_weights)
        self.J[ii, jj] = np.clip(self.J[ii, jj] + np.random.normal(loc=0, scale=settings['mutationSigma']),
                                 -self.max_weights, self.max_weights)
        #Mutation of weights--> mutated weight is generated randomly from scratch

        # MUTATE NEURON BIASES (local field h)
        i = np.nonzero(self.maskh)[0]
        randindex = np.random.randint(0, len(i))
        ii = i[randindex]
        self.h[ii] = np.random.uniform(-1, 1)

        # MUTATE LOCAL TEMPERATURE
        if settings['mutateB']:
            deltaB = np.abs(np.random.normal(1, settings['sigB']))
            self.Beta = self.Beta * deltaB  #TODO mutate beta not by multiplying? How was Beta modified originally?



class food():
    def __init__(self, settings):
        self.xpos = uniform(settings['x_min'], settings['x_max'])
        self.ypos = uniform(settings['y_min'], settings['y_max'])
        self.energy = settings['food_energy']

    def respawn(self, settings):
        self.xpos = uniform(settings['x_min'], settings['x_max'])
        self.ypos = uniform(settings['y_min'], settings['y_max'])
        self.energy = settings['food_energy']

# ------------------------------------------------------------------------------+
# ------------------------------------------------------------------------------+
# --- FUNCTIONS ----------------------------------------------------------------+
# ------------------------------------------------------------------------------+
# ------------------------------------------------------------------------------+


def dist(x1, y1, x2, y2):
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def pdistance_pairwise(x0, x1, dimensions, food=False):
    '''
    Parameters
    ----------
    x0, x1:
        (vectorized) list of coordinates. Can be N-dimensional. e.g. x0 = [[0.5, 2.], [1.1, 3.8]].

    dimensions:
        size of the bounding box, array of length N. e.g. [8., 8.], [xmax - xmin, ymax - ymin].

    food:
        boolean signifying if the distance calculations are between organisms or between organisms and food. In the
        latter case we don't need to compare it both ways around, in the former, theta_mat is a non-symmetric matrix.

    Returns
    -------

    dist_mat:
        upper triangle matrix of pairwise distances accounting for periodic boundaries

    theta_mat:
        full matrix of angles between each position accounting for periodic boundaries
    '''


    # get all unique pairs combinations
    N1 = len(x0)
    N2 = len(x1)

    if food:
        combo_index = list(product(np.arange(N1), np.arange(N2)))
    else:
        if not len(x0) == len(x1):
            raise Exception('x0.shape[0] not equal to x1.shape[0] when comparing organisms.')
        combo_index = list(combinations(np.arange(N1), 2))


    Ii = np.array([x0[i[0]] for i in combo_index])
    Ij = np.array([x1[i[1]] for i in combo_index])

    # calculate distances accounting for periodic boundaries
    # delta = np.abs(Ipostiled_seq - Ipostiled)
    delta = Ij - Ii
    delta = np.where(np.abs(delta) > 0.5 * dimensions, delta - np.sign(delta)*dimensions, delta)

    dist_vec = np.sqrt((delta ** 2).sum(axis=-1))
    theta_vec_ij = np.degrees(np.arctan2(delta[:, 1], delta[:, 0]))  # from org i to org j
    if not food:
        theta_vec_ji = np.degrees(np.arctan2(-delta[:, 1], -delta[:, 0])) # from org j to org i

    if food:
        dist_mat = dist_vec.reshape(N1, N2)
    else:
        dist_mat = np.zeros((N1, N2))
    theta_mat = np.zeros((N1, N2))

    for ii, ind in enumerate(combo_index):
        i = ind[0]
        j = ind[1]
        # can leave this as upper triangle since it's symmetric
        if not food:
            dist_mat[i, j] = dist_vec[ii]
        # need to get a full matrix since eventually these angles are not symmetric
        theta_mat[i, j] = theta_vec_ij[ii]
        # if comparing org-to-org angles, need the other direction as well
        if not food:
            theta_mat[j, i] = theta_vec_ji[ii]


    return dist_mat, theta_mat

def calc_heading(I, food):
    d_x = food.xpos - I.xpos
    d_y = food.ypos - I.ypos
    theta_d = degrees(atan2(d_y, d_x)) - I.r
    theta_d %= 360

    # keep the angles between -180:180
    if theta_d > 180:
        theta_d -= 360
    return theta_d


# Transform bool array into positive integer
def bool2int(x):
    y = 0
    for i, j in enumerate(np.array(x)[::-1]):
        y += j * 2 ** i
    return int(y)


# Transform positive integer into bit array
def bitfield(n, size):
    x = [int(x) for x in bin(int(n))[2:]]
    x = [0] * (size - len(x)) + x
    return np.array(x)

def extract_plot_information(isings, foods):
    isings_info = []
    foods_info = []
    for I in isings:
        isings_info.append([I.xpos, I.ypos, I.r, I.energy])
    for f in foods:
        foods_info.append([f.xpos, f.ypos])
    return isings_info, foods_info




def TimeEvolve(isings, foods, settings, folder, rep):

    # if settings['energy_model']:
    #     for I in isings:
    #         I.energies = []  # Clear .energies, that .avg_energy is calculated from with each iteration
    #         I.energy = settings['initial_energy']  # Setting initial energy

    for I in isings:
        I.reset_state(settings)
        I.generation = rep

    T = settings['TimeSteps']
    # for I in isings:
    #     I.position = np.zeros((2, T))

    # Main simulation loop:
    if settings['plot'] == True:
        #plt.clf()
        # plt.ion()
        fig, ax = plt.subplots()
        #fig.set_size_inches(15, 10)
        isings_all_timesteps = []
        foods_all_timesteps = []
        #artists_all_TS = np.zeros(T)
        #artist_list = []



    '''
    !!! iterating through timesteps
    '''
    if settings['parallel_computing'] and not settings['BoidOn'] and not settings['ANN']:
        pool = mp.Pool(12)

        for t in tqdm(range(T)):

            for I in isings:
                I.UpdateSensors(settings) # update sensors at beginning

            vars_list = [(settings, I.Ssize, I.size, I.s, I.h, I.J, I.Beta) for I in isings]
            s_list = pool.map(parallelizedSequGlauberStep, vars_list)

            for i, I in enumerate(isings):
                I.s = s_list[i]
                I.Move(settings)  # move organism at end

        pool.close()
        pool.join()
    else:
        for t in tqdm(range(T)):
            # print('\r', 'Iteration {0} of {1}'.format(t, T), end='') #, end='\r'
            # print('\r', 'Tstep {0}/{1}'.format(t, T), end='')  # , end='\r'
            if settings['seasons'] == True:
                foods = seasons(settings, foods, t, T)

            # PLOT SIMULATION FRAME
            if settings['plot'] == True and (t % settings['frameRate']) == 0:
                isings_info, foods_info = extract_plot_information(isings, foods)
                isings_all_timesteps.append(isings_info)
                foods_all_timesteps.append(foods_info)

            interact(settings, isings, foods)

            if settings['BoidOn']:
                boid_update(isings, settings)
                for I in isings:
                    I.position[:, t] = [I.xpos, I.ypos]

            else:
                if settings['ANN']:
                    for I in isings:
                        I.ANNUpdate(settings)
                else:
                    for I in isings:
                        I.SequentialGlauberStep(settings)
                
            
            
    if settings['plot']:
        #plotting.animate_plot(artist_list, settings, ax, fig)
        # try:
        print('Creating animation...')
        plotting.animate_plot_Func(isings_all_timesteps, foods_all_timesteps, settings, ax, fig, rep, t, folder)
        # except Exception:
        #     print('There occurred an error during animation...the simulation keeps going')

        '''
        for I in isings:
            if settings['ANN']:
                I.ANNUpdate(settings)
            else:
                I.SequentialGlauberStep(settings)
            I.position[:, t] = [I.xpos, I.ypos]
        '''
'''
#Helper functions parallelization
def parallelSequGlauberStep(I, settings):
    # I = copy.deepcopy(I)
    I.SequentialGlauberStep()
    return I
'''
def parallelizedSequGlauberSteps(isings, settings, asynchronous = False):

    if not asynchronous:

        if settings['cores'] == 0:
            pool = mp.Pool(mp.cpu_count() - 1)
        else:
            pool = mp.Pool(settings['cores'])

        if not asynchronous:
            for I in isings:
                I.UpdateSensors(settings) # update sensors at beginning
                # pass_vars = (settings, I.Ssize, I.size, I.s, I.h, I.J, I.Beta)
                # pool.apply_async(parallelizedSequGlauberStep, args=(pass_vars), callback=collect_result)
            vars_list = [(settings, I.Ssize, I.size, I.s, I.h, I.J, I.Beta) for I in isings]
            s_list = pool.map(parallelizedSequGlauberStep, vars_list)
            pool.close()
            #pool.join()

            # = results

            for i, I in enumerate(isings):
                I.s = s_list[i]
                I.Move(settings)  # move organism at end
        else:
            for I in isings:
                I.UpdateSensors(settings)  # update sensors at beginning
            # pass_vars = (settings, I.Ssize, I.size, I.s, I.h, I.J, I.Beta)
            # pool.apply_async(parallelizedSequGlauberStep, args=(pass_vars), callback=collect_result)
            vars_list = [(settings, I.Ssize, I.size, I.s, I.h, I.J, I.Beta) for I in isings]
            s_list = pool.map_async(parallelizedSequGlauberStep, vars_list)
            pool.close()
            pool.join()



def parallelizedSequGlauberStep(pass_vars):
    settings, Ssize, size, s, h, J, Beta = pass_vars
    thermalTime = int(settings['thermalTime'])

    # update all other neurons a bunch of times
    for j in range(thermalTime):
        perms = np.random.permutation(range(Ssize, size))
        for i in perms:
            s_fac = GlauberStepParallel(i, s, h, J, Beta, size)
            s[i] = s[i] * s_fac
    return s


def GlauberStepParallel(i, s, h, J, Beta, size):
    eDiff = 2 * s[i] * (h[i] + np.dot(J[i, :] + J[:, i], s))
    if Beta * eDiff < np.log(1.0 / np.random.rand() - 1):  # Glauber
        return -1
    else:
        return 1



def collect_result(result):
    global results
    results.append(result)

def TimeEvolve2(isings, BetaFactor, settings, T):
    '''
    Used to calculae heat capacity in natural environment... GA is removed for thid pupose ????
    '''


    #--- POPULATE THE ENVIRONMENT WITH FOOD ---------------+
    foods = []
    for i in range(0, settings['food_num']):
        foods.append(food(settings))

    # INITIALIZATIONS
    BetaOG = []
    for ii, I in enumerate(isings):
        # initialize variables
        I.pm = np.zeros((2, 1)) # position
        I.p2m = np.zeros((2, 1))

        I.Em = 0 # np.zeros((1, T)) # mean Energy
        I.E2m = 0 # np.zeros((1, T)) # mean Energy ** 2

        I.m = 0
        I.m2 = 0

        # set temperature
        BetaOG.append(I.Beta)
        I.Beta = I.Beta * BetaFactor  # scale org's local temperature by the Beta factor

        # random config
        I.randomize_position(settings)
        I.randomize_state()

    # TIME EVOLUTION / MEASUREMENTS
    print('Thermalizing...')
    for t in range(int(T / 10)):
        interact(settings, isings, foods) # interact with food, update raw sensor values.
        I.SequentialGlauberStep(settings)

    print('Beginning measurements...')

    if settings['plot'] == True:
        plt.clf()
        plt.ion()
        fig, ax = plt.subplots()
        folder = 'test'
        rep = 0
        # fig.set_size_inches(15, 10)

    for t in range(T):
        # if (t % 50) == 0 or t == 0:
        #     print(t)

        interact(settings, isings, foods)  # interact with food, update raw sensor values.

        if settings['plot'] == True and (t % settings['frameRate']) == 0:
            plot_frame(settings, folder, fig, ax, isings, foods, t, rep)
            plt.pause(1e-5)
            plt.draw()
            plt.cla()
        if settings['diagnostics'] == True and (t % settings['frameRate']) == 0:
            fitness = 0
            for I in isings:
                fitness += I.food
            print('t : %4i | F: %4f' % (t, fitness))

        for I in isings:
            I.SequentialGlauberStep(settings) # download sensor data into neurons, think (therm.), move.

            # don't include sensor neurons in measurements
            # sus M
            I.m += np.sum(I.s[I.Ssize:]) / float(T)
            I.m2 += np.sum(I.s[I.Ssize:]) ** 2 / float(T)

            # sus p
            p = [I.xpos, I.ypos]
            I.pm[:, 0] += np.divide(p, float(T))
            I.p2m[:, 0] += np.power(p, 2) / float(T)

            # heat spec.
            E = -(np.dot(I.s[I.Ssize:], I.h[I.Ssize:]) +
                  np.dot(np.dot(I.s[I.Ssize:], I.J[I.Ssize:, I.Ssize:]), I.s[I.Ssize:]))
            I.Em += E / float(T)
            I.E2m += E ** 2 / float(T)

    # reset betas back to original values in case of iterated use
    for ii, I in enumerate(isings):
        I.Beta = BetaOG[ii]

    return isings

# Dynamical Critical Learning Algorithm for poising units in a critical state
def HomeostaticGradient(isings, foods, settings, folder, rep):
    T = settings['TimeSteps']
    for I in isings:
        I.m = np.zeros(I.size)
        I.c = np.zeros((I.size, I.size))
        I.C = np.zeros((I.size, I.size))
        # I.var = np.zeros(I.size)

        I.position = np.zeros((2, T))

    # Main simulation loop:
    if settings['plot'] == True:
        plt.clf()
        plt.ion()
        fig, ax = plt.subplots()
        # fig.set_size_inches(15, 10)
    # start_time = time.time()
    for t in range(T):
        # print('Time = ' + str(t))
        # PLOT SIMULATION FRAME
        if settings['plot'] == True and (t % settings['frameRate']) == 0:
            plot_frame(settings, folder, fig, ax, isings, foods, t, rep)
            plt.pause(1e-5)
            plt.draw()
            plt.cla()

        # check_eat_food(settings, isings, foods)
        # calc_closest_food(isings, foods)
        interact(settings, isings, foods)

        for I in isings:
            I.SequentialGlauberStep(settings)
            I.position[:, t] = [I.xpos, I.ypos]
            I.m += I.s

            for iS in range(I.size):
                I.c[iS, iS + 1:] += I.s[iS] * I.s[iS + 1:]
                # I.c[iS, iS] += I.s[iS] ** 2
    # print('Time Elapsed: ' + str(int(time.time() - start_time)))

        # print(isings[0].position[:,t])

    # CALCULATE CORRELATIONS AND MEAN ACTIVATIONS TO GET: dh, dJ
    for I in isings:

        I.m /= T
        I.c /= T

        for iS in range(I.size):
            I.C[iS, iS + 1:] = I.c[iS, iS + 1:] - I.m[iS] * I.m[iS + 1:]
            # I.var[iS] = I.c[iS, iS] - I.m[iS] ** 2

        # resort correlations C1 to match the order of that of the actual C
        # TODO: Does this actually work the way it's supposed to?
        # TODO: Multiply by the sign?
        # TODO: frustrated critical ising models?
        I.sort_critical_correlations()

        I.dh = I.m1 - I.m
        I.dJ = I.C1 - I.C
        # I.dvar = I.var - 1

        I.dh[~I.maskh] = 0
        # I.dvar[~I.maskh] = 0
        I.dJ[~I.maskJ] = 0

def EvolutionLearning(isings, foods, settings, Iterations = 1):
    '''
    Called by "train"
    '''
    #Add command line input to folder name
    s = sys.argv[1:]
    command_input = '_'.join([str(elem) for elem in s])
    folder = 'save/sim-' + time.strftime("%Y%m%d-%H%M%S") + command_input + '/'
    if settings['save_data'] == True:#
        '''
        name save folder
        '''

        if not os.path.exists(folder):
            os.makedirs(folder)
            os.makedirs(folder + 'isings')
            os.makedirs(folder + 'stats')
            os.makedirs(folder + 'figs')

            #save settings dicitionary
            save_settings(folder, settings)
            save_model(folder)
    # else:
    #     folder = None


    count = 0
    for rep in range(Iterations):
        ''' 
        !!! jede Iteration
        '''
        if rep in settings['plot_generations']:
            settings['plot'] = True
        else:
            settings['plot'] = False

        TimeEvolve(isings, foods, settings, folder, rep)

        fitness, fitness_stat = food_fitness(isings)
        if settings['energy_model']:

            for I in isings:
                I.avg_energy = np.median(I.energies)  # Average or median better?
                I.avg_velocity = I.all_velocity / settings['TimeSteps']
            if settings['energy_is_fitness']:
                eat_rate = np.average([I.avg_energy for I in isings])
            else:
                eat_rate = np.sum(fitness_stat)/settings['TimeSteps']
        else:
            eat_rate = np.sum(fitness_stat) / settings['TimeSteps']

        if settings['plot'] == True:
            plt.clf()

        # mutationrate[0], mutationrate[1] = mutation_rate(isings)

        if rep % settings['evolution_rate'] == 0:


            if settings['mutateB']:
                Beta = []
                for I in isings:
                    Beta.append(I.Beta)

                mBeta = np.mean(Beta)
                stdBeta = np.std(Beta)
                maxBeta = np.max(Beta)
                minBeta = np.min(Beta)

        # save rate equal to evolutation rate
        # TODO: Add eatrate; make this useful
            mutationrate = None
            fitm = None
            fitC = None

            if settings['energy_model'] and settings['energy_is_fitness']:
                fit_func_param_name = 'avg_energy'
            else:
                fit_func_param_name = 'eat_rate'

            if settings['mutateB']:
                print(f'{count} | {fit_func_param_name} {eat_rate:.2f} '
                      f'mean_Beta {mBeta:.3f} std_Beta {stdBeta:.3f} min_Beta {minBeta:.3f} max_Beta {maxBeta:.3f}')
            else:
                print(f'{count} | {fit_func_param_name} {eat_rate:.2f}')

            if settings['save_data']:
                if settings['energy_model']:
                    # Clear I.energies in isings_copy before saving
                    # TODO: so if we're not saving, then this list gets infinitely long. Just clear it with/without
                    #  the save condition?
                    isings_copy = deepcopy(isings)
                    for I in isings_copy:
                        I.energies = []

                    save_sim(folder, isings_copy, fitness_stat, mutationrate, fitC, fitm, rep)
                else:
                    save_sim(folder, isings, fitness_stat, mutationrate, fitC, fitm, rep)

        count += 1

        if rep % settings['evolution_rate'] == 0:
            '''
            Irrelevant without critical learning!!
            Evolution via GA! According to evolution rate done every nth iteration
            Does every evolution event represent one generation?
            '''

            isings = evolve(settings, isings, rep)

def CriticalLearning(isings, foods, settings, Iterations=1):
    # settings['TimeSteps'] = 10
    mutationrate = np.zeros(2)

    u = 0.01
    folder = 'save/sim-' + time.strftime("%Y%m%d-%H%M%S") + '/'
    if settings['save_data'] == True:
        if not os.path.exists(folder):
            os.makedirs(folder)
            os.makedirs(folder + 'isings')
            os.makedirs(folder + 'stats')
            os.makedirs(folder + 'figs')
    else:
        folder = None

    count = 0
    l2 = 0.004
    for rep in range(Iterations):
        # CALCULATE self.dh AND self.dJ
        HomeostaticGradient(isings, foods, settings, folder, rep)
        if settings['plot'] == True:
            plt.clf()

        mutationrate[0], mutationrate[1] = mutation_rate(isings)
        fitness, fitness_stat = food_fitness(isings)
        eat_rate = np.sum(fitness_stat)/settings['TimeSteps']

        fitness = (fitness + 1) / 2

        if rep % settings['evolution_rate'] == 0: # save rate linked with evolutation rate
            fitC, fitm, Jmean, hmean = calc_fit(isings, mutationrate, eat_rate, count)

            if settings['mutateB']:
                Beta = []
                for I in isings:
                    Beta.append(I.Beta)

                mBeta = np.mean(Beta)

            if settings['mutateB']:
                print('%4i | F: %4f, T: %1.3f | Fc: %1.3f, Fm: %1.3f | J: %1.3f, H: %1.3f' %
                      (count, eat_rate, mBeta, fitC, fitm, Jmean, hmean))
            else:
                print('%4i | F: %4f | Fc: %1.3f, Fm: %1.3f | J: %1.3f, H: %1.3f' %
                      (count, eat_rate, fitC, fitm, Jmean, hmean))

                # print(count, '|', fitness_stat, fitC, fitm, '|', Jmean, hmean, )
            if settings['save_data'] == True:
                save_sim(folder, isings, fitness_stat, mutationrate, fitC, fitm, rep)

        if rep > settings['TimeStepsGrowth']:
            settings['plot'] = True

        count += 1

        iOrg = 0
        for I in isings:
            # I.J += fitness[iOrg] * (u * I.dJ - l2 * I.J)
            # I.h += fitness[iOrg] * (u * I.dh - l2 * I.h)
            I.J += u * I.dJ - l2 * I.J
            I.h += u * I.dh - l2 * I.h


            Vmax = I.max_weights
            for i in range(I.size):
                if np.abs(I.h[i]) > Vmax:
                    I.h[i] = Vmax * np.sign(I.h[i])
                for j in np.arange(i + 1, I.size):
                    if np.abs(I.J[i, j]) > Vmax:
                        I.J[i, j] = Vmax * np.sign(I.J[i, j])
            iOrg += 1

        if settings['evolution_toggle'] == True:
            if rep % settings['evolution_rate'] == 0:
                isings = evolve(settings, isings, rep)
        else:
            for I in isings:
                I.food = 0




def calc_fit(isings, mutationrate, fitness_stat, count):
    fitC = []
    fitm = []
    Jmean = []
    hmean = []
    for I in isings:
        fitC.append(np.mean(np.abs(
            I.C1[I.maskJ] - I.C[I.maskJ])))

        fitm.append(np.mean(np.abs(I.m1[I.Ssize:] - I.m[I.Ssize:])))

        Jmean.append(np.max(np.abs(I.J)))
        hmean.append(np.max(np.abs(I.h[I.Ssize:])))
    fitC = np.max(fitC)
    fitm = np.mean(fitm)
    fit = fitC + fitm
    Jmean = np.mean(Jmean)  # average of the highest J values
    hmean = np.mean(hmean)  # average of the highest h values

    return fitC, fitm, Jmean, hmean

def food_fitness(isings):
    fitness = []
    for I in isings:

        fitness.append(I.food)

    fitness = np.array(fitness, dtype='float')
    mask = fitness != 0

    fitnessN = copy.deepcopy(fitness)
    fitnessN[mask] /= float(np.max(fitnessN))
    # fitness[mask] /= float(np.max(fitness))

    return fitnessN, fitness

def evolve(settings, I_old, gen):
    size = settings['size']
    nSensors = settings['nSensors']
    nMotors = settings['nMotors']
    
    '''
    !!!fitness function!!!
    '''
    if settings['energy_model'] and settings['energy_is_fitness']:
        I_sorted = sorted(I_old, key=operator.attrgetter('avg_energy'), reverse=True)
    else:
        I_sorted = sorted(I_old, key=operator.attrgetter('food'), reverse=True)
    I_new = []

    alive_num = int(settings['pop_size'] - settings['numKill'])
    elitism_num = int(alive_num/2) # only the top half of the living orgs can duplicate

    numMate = int(settings['numKill'] * settings['mateDupRatio'])
    numDup = settings['numKill'] - numMate

    for i in range(0, alive_num):
        I_new.append(I_sorted[i])

    # --- GENERATE NEW ORGANISMS ---------------------------+
    orgCount = settings['pop_size'] + gen * settings['numKill']

    # DUPLICATION OF ELITE POPULATION
    for dup in range(0, numDup):
        candidateDup = range(0, elitism_num)
        random_index = sample(candidateDup, 1)[0]

        name = copy.deepcopy(I_sorted[random_index].name) + 'm'
        I_new.append(ising(settings, size, nSensors, nMotors, name))

        #  TODO: need to seriously check if mutations are occuring uniquely
        # probably misusing deepcopy here, figure this shit out
        I_new[-1].Beta = copy.deepcopy(I_sorted[random_index].Beta)
        I_new[-1].J = copy.deepcopy(I_sorted[random_index].J)
        I_new[-1].h = copy.deepcopy(I_sorted[random_index].h)
        I_new[-1].maskJ = copy.deepcopy(I_sorted[random_index].maskJ)
        # I_new[-1].maskJtriu = I_sorted[random_index].maskJtriu
        
        '''
        only important with critical learning
        '''
        try:
            I_new[-1].C1 = I_sorted[random_index].C1
        except NameError:
            pass

        # MUTATE SOMETIMES
        if np.random.random() < settings['mutationRateDup']:
            I_new[-1].mutate(settings)

        # random mutations in duplication

    # MATING OF LIVING POPULATION (DOUBLE DIPPING ELITE)
    for mate in range(0, numMate):
        # TODO: negative weight mutations?!
        # SELECTION (TRUNCATION SELECTION)
        candidatesMate = range(0, len(I_new)) # range(0, alive_num) to avoid double dipping
        random_index = sample(candidatesMate, 2)
        org_1 = I_sorted[random_index[0]]
        org_2 = I_sorted[random_index[1]]

        # CROSSOVER
        J_new = np.zeros((size, size))
        h_new = np.zeros(size)

        # load up a dummy maskJ which gets updated
        maskJ_new = np.zeros((size, size), dtype=bool)

        crossover_weight = random()

        # CROSS/MUTATE TEMPERATURE
        if settings['mutateB']:
            # folded normal distribution
            deltaB = np.abs( np.random.normal(1, settings['sigB']) )

            Beta_new = ( (crossover_weight * org_1.Beta) + \
                            ((1 - crossover_weight) * org_2.Beta) ) * deltaB
        else:
            Beta_new = org_1.Beta

        # CROSS WEIGHTS
        for iJ in range(0, size):
            crossover_weight = random()

            h_new[iJ] = (crossover_weight * org_1.h[iJ]) + \
                        ((1 - crossover_weight) * org_2.h[iJ])

            for jJ in range(iJ + 1, size):
                crossover_weight = random()

                # check if these hidden neurons are disconnected to begin with
                if org_1.maskJ[iJ, jJ] != 0 and org_2.maskJ[iJ, jJ] != 0:
                    J_new[iJ, jJ] = (crossover_weight * org_1.J[iJ, jJ]) + \
                                    ((1 - crossover_weight) * org_2.J[iJ, jJ])
                    maskJ_new[iJ, jJ] = org_1.maskJ[iJ, jJ]
                elif np.random.randint(2) == 0:
                    J_new[iJ, jJ] = org_1.J[iJ, jJ]
                    maskJ_new[iJ, jJ] = org_1.maskJ[iJ, jJ]
                else:
                    J_new[iJ, jJ] = org_2.J[iJ, jJ]
                    maskJ_new[iJ, jJ] = org_2.maskJ[iJ, jJ]

                if np.abs(J_new[iJ, jJ]) > org_1.max_weights:
                    J_new[iJ, jJ] = org_1.max_weights


        # TODO: include name of parents
        name = 'gen[' + str(gen) + ']-org[' + str(orgCount) + ']'
        I_new.append(ising(settings, size, nSensors, nMotors, name))

        I_new[-1].Beta = Beta_new
        I_new[-1].J = J_new
        I_new[-1].h = h_new
        I_new[-1].maskJ = maskJ_new

        # MUTATE IN GENERAL
        I_new[-1].mutate(settings)

        orgCount += 1

    for I in I_new:
        I.reset_state(settings)

    return I_new

def save_settings(folder, settings):
    with open(folder + 'settings.csv', 'w') as f:
        for key in settings.keys():
            f.write("%s,%s\n" % (key, settings[key]))
    pickle_out = open('{}settings.pickle'.format(folder), 'wb')
    pickle.dump(settings, pickle_out)
    pickle_out.close()

def save_model(folder):
    src = 'embodied_ising.py'
    dst = folder + src
    copyfile(src, dst)


def save_sim(folder, isings, fitness_stat, mutationrate, fitC, fitm, gen):


    filenameI = folder + 'isings/gen[' + str(gen) + ']-isings.pickle'
    filenameS = folder + 'stats/gen[' + str(gen) + ']-stats.pickle'

    if type(mutationrate) is not type(None):
        mutationh = mutationrate[0]
        mutationJ = mutationrate[1]
    else:
        mutationh = None
        mutationJ = None

    pickle_out = open(filenameI, 'wb')
    pickle.dump(isings, pickle_out)
    pickle_out.close()

    pickle_out = open(filenameS, 'wb')

    if type(mutationrate) is not type(None):
        pickle.dump((fitness_stat, (mutationh, mutationh ** 2), (mutationJ, mutationJ ** 2), fitC, fitm), pickle_out)
        pickle_out.close()
    else:
        pickle.dump(fitness_stat, pickle_out)
        pickle_out.close()


def mutation_rate(isings):
    for I in isings:

        hmutation = np.abs(I.dh[I.maskh])
        Jmutation = np.abs(I.dJ[I.maskJ])

    hmutation = np.mean(hmutation)
    Jmutation = np.mean(Jmutation)
    return hmutation, Jmutation

def sigmoid(x):
    y = 1/(1 + np.exp(-x))
    return y

def logit(x):
    y = np.log(x / (1 - x))
    return y

# def check_eat_food(settings, isings, foods):
#     '''
#     Seems not to be used for anything
#     '''
#     for food in foods:
#         for I in isings:
#
#             food_org_dist = dist(I.xpos, I.ypos, food.xpos, food.ypos)
#
#             # EAT/RESPAWN FOOD
#             if food_org_dist <= 0.075:
#                 I.fitness += food.energy
#                 food.respawn(settings)
#
#         # RESET DISTANCE AND HEADING TO NEAREST FOOD SOURCE
#         I.d_food = I.maxRange
#         I.r_food = 0

def calc_closest_food(isings, foods):
    for food in foods:
        for I in isings:

            # CALCULATE DISTANCE TO SELECTED FOOD PARTICLE
            food_org_dist = dist(I.xpos, I.ypos, food.xpos, food.ypos)

            # DETERMINE IF THIS IS THE CLOSEST FOOD PARTICLE
            if food_org_dist < I.d_food:
                I.d_food = food_org_dist
                I.r_food = calc_heading(I, food)

def seasons_func(food_max, food_min, t, year_t):
    curr_t = (t % year_t)
    #Autumn
    if curr_t < (year_t / 2):
        wanted_food_len = int(np.round(food_max + ((food_min - food_max) / 
                                                   (year_t / 2)) * curr_t))
    else:
        wanted_food_len = int(np.round(food_min + ((food_max - food_min) / 
                                                   (year_t / 2)) * (curr_t - (year_t / 2))))
    return wanted_food_len
    
    
                
def seasons(settings, foods, t, T):
    foods = deepcopy(foods)
    years_per_i = settings['years_per_iteration']
    min_food_rel = settings['min_food_winter']
    if min_food_rel > 1 or min_food_rel < 0:
        raise Exception("'min_food_winter' has to be a float between 0 and 1")
    max_food = settings['food_num']
    min_food = int(np.round(max_food * min_food_rel))
    year_t = T / years_per_i #amount of time steps corresponding to half a year
    wanted_food_len = seasons_func(max_food, min_food, t, year_t)
    
    diff_food_len = wanted_food_len - len(foods)
    
    if diff_food_len < 0:
        for i in range(abs(diff_food_len)):
            #rand = np.random.randint(0, len(foods)) Is randomness important here?
            del foods[-1]
    elif diff_food_len > 0:
        for i in range(abs(diff_food_len)):
            foods.append(food(settings))
     
    return foods

#TODO: double check if this is working as intended!
def interact(settings, isings, foods):
    '''
    consider making a matrix of values instead of looping through all organisms
    currently, there is redundancy in these loops which might end up being time consuming
    '''

    # calculate all agent-agent and agent-food distances
    Ipos = np.array( [[I.xpos, I.ypos] for I in isings] )
    foodpos = np.array( [[food.xpos, food.ypos] for food in foods] )
    dimensions = np.array([settings['x_max'] - settings['x_min'], settings['y_max'] - settings['y_min']])
    org_heading = np.array([I.r for I in isings]).reshape(len(Ipos), 1)

    dist_mat_org, theta_mat_org = pdistance_pairwise(Ipos, Ipos, dimensions, food=False)
    dist_mat_food, theta_mat_food = pdistance_pairwise(Ipos, foodpos, dimensions, food=True)

    # calculate agent-agent and agent-food angles
    theta_mat_org = theta_mat_org - org_heading
    theta_mat_food = theta_mat_food - org_heading

    theta_mat_org = np.mod(theta_mat_org, 360)
    theta_mat_org = np.where(theta_mat_org > 180, theta_mat_org - 360, theta_mat_org)

    theta_mat_food = np.mod(theta_mat_food, 360)
    theta_mat_food = np.where(theta_mat_food > 180, theta_mat_food - 360, theta_mat_food)

    # calculate org sensor
    org_sensor = np.where(np.abs(theta_mat_org) > 90, 0, np.cos(np.deg2rad(theta_mat_org)))
    org_radius = np.array([I.radius for I in isings]).reshape(len(Ipos), 1)
    org_sensor = (org_sensor * org_radius) / (dist_mat_org + dist_mat_org.T + 1e-6) ** 2
    np.fill_diagonal(org_sensor, 0)
    org_sensor = np.sum(org_sensor, axis=1)

    for i, I in enumerate(isings):
        if settings['energy_model']:
            I.energies.append(I.energy)

        # threshold food vision
        # visionMask = dist_mat_food < settings['vision_radius']


        minFoodDist = np.min(dist_mat_food[i, :])
        foodInd = np.argmin(dist_mat_food[i, :])
        I.d_food = minFoodDist  # Distance to closest food
        I.r_food = theta_mat_food[i, foodInd] # "angle" to closest food

        # if food is close enough, eat it
        if minFoodDist <= settings['org_radius']:

            # check if food is to be shared/distributed
            if not settings['share_food']:
                I.food += foods[foodInd].energy
                if settings['energy_model']:
                    I.energy += foods[foodInd].energy
            else:
                neighbours = dist_mat_org[i, :] < settings['share_radius']
                neighbours[i] = False
                numneighbours = np.sum(neighbours)

                if numneighbours > 0:  # if there are neighbours, share it

                    # diminishing returns for sharing food among more people
                    foodshare = settings['share_param'] * np.exp(-(numneighbours - 1)/5)

                    I.food += foods[foodInd].energy * \
                              (1 - settings['zerosum_param'] * foodshare * numneighbours)

                    I.foodfound += 1
                    I.foodshared += foods[foodInd].energy * foodshare * numneighbours

                    if settings['energy_model']:
                        I.energy += foods[foodInd].energy * \
                                    (1 - settings['zerosum_param'] * foodshare * numneighbours)

                    for Ineighbour in [II for ii, II in enumerate(isings) if neighbours[ii]]:
                        Ineighbour.food += foods[foodInd].energy * foodshare
                        if settings['energy_model']:
                            Ineighbour.energy += foods[foodInd].energy * foodshare
                        Ineighbour.foodgiven += foods[foodInd].energy * foodshare
                else:  # if no neighbours
                    I.food += foods[foodInd].energy
                    if settings['energy_model']:
                        I.energy += foods[foodInd].energy
                    I.foodfound += 1
            '''
            finess is proportional to energy
            '''
            foods[foodInd].respawn(settings)

        I.org_sens = org_sensor[i]




def update_ising(settings, I, t):
    I.SequentialUpdate(settings)
    I.position[:, t] = [I.xpos, I.ypos]
    I.m += I.s

    for iS in range(I.size):
        I.c[iS, iS + 1:] += I.s[iS] * I.s[iS + 1:]


#####################################
# BOID THINGS
#####################################

def boid_dv(isings, settings):
    vision_radius = 0.75
    N = len(isings)
    combos = list(combinations(range(0, N), r=2))

    # ## CALCULATE CUMULATIVE AVERAGES
    # # t_center = np.zeros(2)
    # # t_velocity = np.zeros(2)
    # pos_list = np.zeros((N, 2))
    # vel_list = np.zeros((N, 2))

    pos_list = np.array([[I.xpos, I.ypos] for I in isings])
    vel_list = np.array([[I.dx, I.dy] for I in isings])

    # for i, I in enumerate(isings):
    #     pos_list[i, :] = np.array((I.xpos, I.ypos))
    #     vel_list[i, :] = np.array((I.dx, I.dy))
    #     # t_center += pos_i
    #     # t_velocity += np.array((I.dx, I.dy))

    ## CALCULATE DISTANCE MATRIX
    # TODO: use the new periodic distance function (and time it).
    dist_mat = np.zeros((len(isings), len(isings), 2))
    for pair in combos:
        ii = pair[0]
        jj = pair[1]

        pos_i = pos_list[ii]
        pos_j = pos_list[jj]

        dist_mat[ii, jj, :] = pointing_vector(pos_i, pos_j, settings)
        dist_mat[jj, ii, :] = -dist_mat[ii, jj, :]

    # for RULE 3: alternate form: (1/d^2 - 1/d^3)

    radius_mat = np.linalg.norm(dist_mat, axis=-1)  # used as a radius to localize global information by weighting
    radius_mat = np.dstack([radius_mat] * 2)
    # radius_mat = np.divide(radius_mat, np.sum(radius_mat, axis=0)).transpose()
    # radius_mat = np.multiply(radius_mat, np.sum(zeronaninf(1 / radius_mat), axis=0)).transpose()
    vision_mat = radius_mat < vision_radius
    # vision_mat = np.dstack([vision_mat] * 2)


    # corr_mat = -1 * np.sign(dist_mat) / np.dstack([np.power(radius_mat, 2)] * 2)
    # corr_mat[np.isinf(corr_mat)] = 0
    # corr_mat[np.isnan(corr_mat)] = 0

    localized_info = {}
    # localized_info['pos'] = np.divide(np.tile(pos_list, (len(isings), 1, 1)), np.dstack([np.power(radius_mat, 2)] * 2))
    # localized_info['corr'] = np.tanh(corr_mat)  # already 1/r^2 dependant
    # localized_info['vel'] = np.divide(np.tile(vel_list, (len(isings), 1, 1)), np.dstack([radius_mat] * 2))

    # pos_mat = np.tile(pos_list, (len(isings), 1, 1))
    # vel_mat = np.tile(vel_list, (len(isings), 1, 1))

    # localized_info['pos'] = np.tile(pos_list, (len(isings), 1, 1))
    localized_info['corr'] = np.divide(-1 * dist_mat, np.power(radius_mat, 2))
    localized_info['vel']  = np.tile(vel_list, (len(isings), 1, 1))


    for key, value in localized_info.items():
        new_value = zeronaninf(value)
        # new_value = np.tanh(value)
        localized_info.update({key: new_value})

    # corr_mat = np.tanh(corr_mat)

    ## CALCULATE INDIVIDUAL BOID FORCES
    for i, Ii in enumerate(isings):
        position_i = np.array((Ii.xpos, Ii.ypos))
        velocity_i = np.array((Ii.dx, Ii.dy))

        # RULE 1: percieved center
        # p_center_i = t_center - position_i
        # p_center_i = np.sum(localize_boid_info(pos_list, radius_mat[i, :]), axis=0)
        # p_center_i = (np.sum(
        #     localized_info['pos'][i, :, :] * vision_mat[i, :, :], axis=0) /
        #               np.sum(vision_mat[i, :, :], axis=0))

        cohesion_vector = (np.sum(dist_mat[i, :, :] * vision_mat[i, :, :], axis=0) /
                           np. sum(vision_mat[i, :, :], axis=0))

        # RULE 2: percieved correction
        # p_corr_i = np.sum(corr_mat[i, :, :], axis=0)
        # p_corr_i = np.sum(localize_boid_info(corr_mat[i, :, :], radius_mat[i, :]), axis=0)
        repulsion_vector = (np.sum(
            localized_info['corr'][i, :, :] * vision_mat[i, :, :], axis=0) /
                   np.sum(vision_mat[i, :, :], axis=0))

        # RULE 3: percieved velocity
        # p_velocity_i = t_velocity - velocity_i
        # p_velocity_i = np.sum(localize_boid_info(vel_list, radius_mat[i, :]), axis=0)
        p_velocity_i = (np.sum(
            localized_info['vel'][i, :, :] * vision_mat[i, :, :], axis=0) /
                       np.sum(vision_mat[i, :, :], axis=0))

        # average
        # p_center_i /= (N - 1)
        # p_corr_i /= (N - 1)
        # p_velocity_i /= (N - 1)
        random_dv = np.random.normal(size=2)

        # dv1 = p_center_i - position_i
        # dv2 = p_corr_i
        # dv3 = p_velocity_i - velocity_i

        # dv1 = p_center_i - position_i
        # dv2 = p_corr_i
        # dv3 = p_velocity_i - velocity_i

        # update
        # dv1 = 1 * (p_center_i - position_i)
        # dv2 = 0.1 * p_corr_i
        dv3 = 1 * (p_velocity_i - velocity_i)
        # dv4 = 0.001 * random_dv
        Ii.dvb = cohesion_vector + (0.2 * repulsion_vector) + dv3 + (0.005 * random_dv)

def boid_move(agents, settings):
    for I in agents:
        # dvb = I.dv1 + I.dv2 + I.dv3 + I.dv4
        # normdvb = np.linalg.norm(dvb)
        # if normdvb > settings['dv_max']:
        #     dvb /= normdvb
        #     I.dvb = dvb * settings['dv_max']
        # else:
        #     I.dvb = dvb

        I.v += I.dvb * settings['dt']
        I.r = np.arctan2(I.v[1], I.v[0]) * 180 / np.pi

        mag_v = np.sqrt(np.sum(np.power(I.v, 2)))

        if mag_v > settings['v_max']:
            I.v /= mag_v / settings['v_max']

        I.dx = I.v[0] * settings['dt']
        I.dy = I.v[1] * settings['dt']
        I.xpos += I.dx
        I.ypos += I.dy

        # torus boundary conditions
        I.xpos = (I.xpos + settings['x_max']) % settings['x_max']
        I.ypos = (I.ypos + settings['y_max']) % settings['y_max']

# Derivative of neuron output
def transfer_derivative(output):
    return (1 - np.power(np.tanh(output), 2))

# Backpropogate error and update neuron weights
# def boid_backprop_error(isings):
#
#     for I in isings:
#         I.error =


# def boid_learn(isings):
#     boid_dv(isings)  # calculate boid dv
#     boid_backprop_error(isings)  # calculate error w.r.t. boid activations

def boid_update(isings, settings):
    boid_dv(isings, settings)  # Calculate boid dv
    boid_move(isings, settings)  # Apply boid dv

def localize_boid_info(global_vec, dist_vec):
    normalized = np.divide(global_vec, np.transpose(np.tile(dist_vec, (2,1))))
    normalized = zeronaninf(normalized)
    return normalized

def zeronaninf(mat):
    mat[np.isinf(mat)] = 0
    mat[np.isnan(mat)] = 0
    return mat

def pointing_vector(pos_i, pos_j, settings):
    L = [settings['x_max'] - settings['x_min'], settings['y_max'] - settings['y_min']]
    diff = pos_j - pos_i
    for dim in range(len(L)):
        diff[dim] = (diff[dim] +
                     (np.abs(diff[dim]) > L[dim] / 2) * -1 * (1 - 2 * (diff[dim] < L[dim] / 2)) * L[dim])
    return diff
# def boid_error(isings):
#
#
# def af_derivative(self):

