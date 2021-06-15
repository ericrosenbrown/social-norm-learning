from functools import reduce
import copy
from .world import Worlds
import torch
import numpy as np

class Environment:

    def __init__(self, world, start, viz_start, categories):
        self.world = world
        self.state_start = start
        self.viz_start = viz_start
        self.categories = categories
        self.ncategories = len(categories)
        self.actions = [(-1,0), (0,1), (1,0), (0,-1), (0,0)]
        self.act_map = Worlds.act_map
        self.act_name = Worlds.act_name
        self.dtype = torch.float32

    def get_start_state(self):
        return copy.copy(self.state_start)

    def get_single_timestep_action(self):
        """
        Retrieves a single discrete timestep action from a human
        """
        k = input("Action: ")
        if k =="w":
            return 0
        elif k =="d":
            return 1
        elif k == "s":
            return 2
        elif k =="a":
            return 3
        elif k == "stop":
            return -1
        else:
            return 4

    def acquire_human_demonstration(self, max_length=15):
        """
        Acquires a trajectory of actions from a human
        of length of at least 1, starting from an initial state
        """
        cur_state = self.get_start_state()
        self.visualize_environment(cur_state[0],cur_state[1])
        action = self.get_single_timestep_action()
        action_sequence = [action]
        ## Create a clone of the start state and allow the human to demonstrate from there
        ## Update the current state
        cur_state = [a+b for a,b in zip(cur_state, self.act_map[action])]
        print(f"Action({action})={self.act_name[action]}")

        for step in range(max_length):
            self.visualize_environment(cur_state[0],cur_state[1])
            action = self.get_single_timestep_action()
            if action == -1:
                break
            action_sequence.append(action)
            cur_state = [a+b for a,b in zip(cur_state, self.act_map[action])]
            print(f"Action({action})={self.act_name[action]}")

        trajcoords = reduce((lambda seq, a: seq + [[seq[len(seq)-1][0] + self.actions[a][0], seq[len(seq)-1][1] + self.actions[a][1]]]), action_sequence, [self.state_start])
        return action_sequence, trajcoords

    def get_matmap(self):
        """
        Return binarized form of a grid map:
        (rows, cols, category) -- true false for the category
        """
        r,c = self.world.shape
        shape = (r,c,len(self.categories))
        matmap = np.zeros(shape)
        for k in self.categories:
            matmap[:,:,k] = 0 + (self.world == k)
        return torch.tensor(matmap, dtype=self.dtype, requires_grad=False)

    def get_mattrans(self):
        """
        Return transition matrix
        """
        def clip(v,min,max):
            if v < min: v = min
            if v > max-1: v = max-1
            return(v)
        nrows,ncols = self.world.shape
        nacts = len(self.actions)
        mattrans = np.zeros((nacts, nrows*ncols, nrows*ncols))
        for acti in range(nacts):
            act = self.actions[acti]
            for i1 in range(nrows):
                for j1 in range(ncols):
                    inext = clip(i1 + act[0],0,nrows)
                    jnext = clip(j1 + act[1],0,ncols)
                    for i2 in range(nrows):
                        for j2 in range(ncols):
                            mattrans[acti,i1*ncols+j1,i2*ncols+j2] = 0+((i2 == inext) and (j2 == jnext))
        return torch.tensor(mattrans, dtype=self.dtype, requires_grad=False)

    def visualize_environment(self, robox, roboy):
        print("===================================")
        for r in range(len(self.world)):
            rowstr = ""
            for c in range(len(self.world[r])):
                if r==robox and c==roboy:
                    rowstr += "R"
                else:            
                    rowstr += str(self.world[r][c])
            print(rowstr)
