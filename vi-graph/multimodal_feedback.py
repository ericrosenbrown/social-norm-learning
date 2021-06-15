import random
import copy
from env.environment import Environment
from env.world import Worlds
from numpy.random import choice
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class ComputationGraph:
    def __init__(self, env):
        """
        Initialize the computation graph

        The aim of the computation graph is to learn
        R(s,a,s')
        """
        ## Environment consist of the grid world and categories
        self.env = env
        self.dtype = env.dtype
        ## Category to location map
        self.matmap = env.get_matmap()
        ## Transition matrix P
        self.mattrans = env.get_mattrans()
        ## Learning rate alpha
        self.learning_rate = 0.001
        ## Discount factor
        self.gamma = 0.90
        ## Boltzmann temperature for Q softmax
        self.beta = 10.0
        ## Horizon - number of timesteps to simulate into the future from the current point
        self.horizon = 50
        ## Softmax Function, along dimension 1
        self.softmax = torch.nn.Softmax(dim=1)
        ## Number of gradient updates to compute
        self.num_updates = 100
        
        #initial random guess on r
        #r starts as [n] categories. we will inflate this to [rows,cols,n]
        # to multiple with matmap and sum across category
        #r = np.random.rand(5)*2-1
        ## Reward function as a function of category
        ## NOTE: Initialize reward to a constant value because a policy can
        # be the optimal under any constant reward function. Therefore,
        # initialize  r to 0 everywhere
        r = np.zeros(env.ncategories)
        self.rk = torch.tensor(r, dtype=self.dtype, requires_grad=True)

    def forward(self):
        """
        This is the planning function. For the agent's estimation of the reward function,
        what are the estimated Q values for taking each action, and what is the corresponding stochastic policy?
        The stochastic policy is computed using the softmax over the Q values.
        """
        ## Expand the reward function to map onto the matrix map
        new_rk = self.rk.unsqueeze(0)
        new_rk = new_rk.unsqueeze(0)
        nrows, ncols, ncategories = self.matmap.shape
        new_rk = new_rk.expand(nrows, ncols, ncategories)
        ## Dot product to obtain the reward function applied to the matrix map
        rfk = torch.mul(self.matmap, new_rk)
        rfk = rfk.sum(axis=-1)
        rffk = rfk.view(nrows*ncols) ## 1D view of the 2D grid
        #initialize the value function to be the reward function, required_grad should be false
        v = rffk.detach().clone()

        #inflate v to be able to multiply mattrans
        v = v.unsqueeze(0)
        v = v.expand(nrows*ncols,nrows*ncols)

        ## NOTE: Does the concept of a goal exist in this case? Agent isn't aware of the goal state
        ##       and the environment doesn't explicitly specify the goal state
        ## NOTE: Do rollouts to a specified horizon
        for _ in range(self.horizon):
            ## Compute Q values
            ## This forces Q to be (nrows*ncols x nacts)
            Q = torch.mul(self.mattrans, v).sum(dim=-1).T
            pi = self.softmax(self.beta * Q)
            next_Q = (Q * pi).sum(dim=1)
            v = rffk + self.gamma * next_Q
        return pi, Q

    def loss(self, pi, Q, trajacts, trajcoords):
        nrows, ncols = self.env.world.shape
        loss = 0
        for i in range(len(trajacts)):
            acti = trajacts[i]
            state = trajcoords[i]
            loss += -torch.log(pi[state[0]*ncols+state[1]][acti])
        loss.backward()
        return loss

    def update(self, trajacts, trajcoords):
        print("LEARNING!***************")
        piout, Qout, loss = None, None, None
        for k in range(self.num_updates):
            piout, Qout = self.forward()
            loss = self.loss(piout, Qout, trajacts, trajcoords)
            with torch.no_grad():
                grads_value = self.rk.grad
                self.rk -= self.learning_rate * grads_value
                self.rk.grad.zero_()
        return piout, Qout, loss, self.rk

    ## TODO: Finish findpol and plotpolicy

    def plotpolicy(self, pi):
        def findpol(grid,pi,r,c):
            if grid[r][c] != 6: return
            maxprob = max(pi[r*ncols+c,:])
            a = 6
            for ana in range(5):
                if pi[r*ncols+c, ana] == maxprob: a = ana
            grid[r][c] = a
            r += self.env.actions[a][0]
            c += self.env.actions[a][1]
            findpol(grid,pi,r,c)

        startr, startc = self.env.viz_start[0], self.env.viz_start[1]
        nrows, ncols = self.env.world.shape
        grid = []
        for r in range(nrows):
            line = []
            for c in range(ncols):
                line += [self.env.world[r][c]+6]
            grid += [line]
        findpol(grid,pi,startr,startc)
        for r in range(nrows):
            line = ""
            for c in range(ncols):
                line += '^>v<x?01234'[grid[r][c]]
            print(line)

def train(episodes, cg):
    """
    Train the agent's estimate of the reward function
    """
    tensor_time = []
    for ep in range(episodes):
        #Collect Trajectories
        trajacts, trajcoords = cg.env.acquire_human_demonstration(max_length=15)
        ## Update the reward function based on the trajectory
        pi, Q, loss, rk = cg.update(trajacts, trajcoords)
        tensor_time.append(copy.deepcopy(rk))
        if ep % 1 == 0:
            print(tensor_time)
            cg.plotpolicy(pi)
        
if __name__ == '__main__':
    ## Select a world
    idx = 0
    grid_maps, state_starts, viz_starts = Worlds.define_worlds()
    env = Environment(grid_maps[idx], state_starts[idx], viz_starts[idx], Worlds.categories)
    cg = ComputationGraph(env)
    episodes = 10

    train(episodes, cg)
