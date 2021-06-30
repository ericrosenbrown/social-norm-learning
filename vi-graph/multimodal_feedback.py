import random
import copy
import pickle
import datetime
import argparse
import os.path
from collections import namedtuple
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

    def set_reward_estimate(self, r):
        self.rk = torch.tensor(r, dtype=self.dtype, requires_grad=True)

    def current_reward_estimate(self):
        return copy.deepcopy(self.rk.cpu().detach().numpy())

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

    def action_loss(self, pi, Q, trajacts, trajcoords):
        nrows, ncols = self.env.world.shape
        loss = 0
        for i in range(len(trajacts)):
            acti = trajacts[i]
            state = trajcoords[i]
            loss += -torch.log(pi[state[0]*ncols+state[1]][acti])
        loss.backward()
        return loss

    def action_update(self, trajacts, trajcoords):
        print("LEARNING!***************")
        piout, Qout, loss = None, None, None
        for k in range(self.num_updates):
            piout, Qout = self.forward()
            loss = self.action_loss(piout, Qout, trajacts, trajcoords)
            with torch.no_grad():
                grads_value = self.rk.grad
                self.rk -= self.learning_rate * grads_value
                self.rk.grad.zero_()
        return piout, Qout, loss, self.rk

    def scalar_loss(self, pi, Q, trajacts, trajcoords, scalar):
        nrows, ncols = self.env.world.shape
        loss = 0
        for i in range(len(trajacts)):
            acti = trajacts[i]
            state = trajcoords[i]
            loss += -torch.log(pi[state[0]*ncols+state[1]][acti])
        loss.backward()
        return loss

    def scalar_scale(self, pi, action, r,c, scalar):
        def clip(v, nmin, nmax):
            if v < nmin: v = nmin
            if v > nmax-1: v = nmax-1
            return(v)

        nrows, ncols = self.env.world.shape
        scale = clip(scalar / pi[r*ncols + c][action].item(), -1, 2)
        return scale

    def scalar_update(self, scalar, action, r, c):
        print("LEARNING!***************")
        piout, Qout, loss = None, None, None
        ## NOTE: Scalar feedback performs a SINGLE on-policy update
        piout, Qout = self.forward()
        loss = self.scalar_loss(piout, Qout, [action], [(r,c)], scalar)
        scale = self.scalar_scale(piout, action, r, c, scalar)
        with torch.no_grad():
            grads_value = self.rk.grad
            self.rk -= (self.learning_rate * grads_value) * scale
            self.rk.grad.zero_()
        return piout, Qout, loss, self.rk

    ## TODO: Finish findpol and plotpolicy

    def plotpolicy(self, pi):
        def findpol(grid,pi,r,c, iteration):
            ## Commented out to allow agent to roam anywhere
            #if grid[r][c] != 6: return
            iteration += 1
            if grid[r][c] == 10: return
            maxprob = max(pi[r*ncols+c,:])
            a = 6
            for ana in range(5):
                if pi[r*ncols+c, ana] == maxprob: a = ana
            grid[r][c] = a
            r += self.env.actions[a][0]
            c += self.env.actions[a][1]
            if iteration < 989:
                findpol(grid,pi,r,c,iteration)
            else:
                print(f'Exceeded iterations')
                return

        startr, startc = self.env.viz_start[0], self.env.viz_start[1]
        nrows, ncols = self.env.world.shape
        grid = []
        iteration = 0
        for r in range(nrows):
            line = []
            for c in range(ncols):
                line += [self.env.world[r][c]+6]
            grid += [line]
        findpol(grid,pi,startr,startc, iteration)
        for r in range(nrows):
            line = ""
            for c in range(ncols):
                line += '^>v<x?01234'[grid[r][c]]
            print(line)

    def chooseAction(self, r, c):
        pi, Q = self.forward()
        epsilon = 0.25
        nrows, ncols, ncategories = self.matmap.shape
        action_prob = pi[r*ncols+c].cpu().detach().numpy()
        print("Original Action Probabilities (Up, Right, Down, Left, Stay): ")
        print(np.round(action_prob, 3))
        ## Filter out invalid actions
        if r == 0:
            action_prob[0] = 0
        if c == ncols - 1:
            action_prob[1] = 0
        if r == nrows - 1:
            action_prob[2] = 0
        if c == 0:
            action_prob[3] = 0

        ## Renormalize probabilities
        action_prob = action_prob / np.sum(action_prob)
        print("Action Probabilities (Up, Right, Down, Left, Stay): ")
        print(np.round(action_prob, 3))

        ## Epsilon-greedy (currently disabled)
        # r = random.uniform(0, 1)
        # print("Random Number: " + str(r))

        # if r < epsilon:
        #   permitable_actions = np.nonzero(action_prob)[0]
        #   choice = np.random.choice(permitable_actions, 1)[0]
        #   print("Picking a random action...")
        #   print(choice)
        #   return choice
        # print("Picking from probabilities...")
        # choice = np.random.choice(5, 1, p=action_prob)[0]
        # print(choice)
        choice = np.argmax(action_prob)
        return choice, copy.deepcopy(pi[r*ncols+c].cpu().detach().numpy())

Transition = namedtuple("Transition", "t reward_estimate state action policy feedback_type feedback_value")

class Trial:
    def __init__(self, initial_state, goal, reward_estimate, grid_map):
        self.initial_state = initial_state
        self.goal = goal
        self.reward_estimate = reward_estimate
        self.grid_map = grid_map
        self.transitions = []

    def add_transition(self, t, r, s, a, p, ft, fv):
        self.transitions.append(Transition(t, r, s, a, p, ft, fv))

def train(episodes, cg):
    """
    Train the agent's estimate of the reward function
    """
    tensor_time = []
    pi, Q, loss, rk = None, None, None, None
    max_steps = 100
    all_trial_data = []
    for ep in range(episodes):
        ## Initialize episode:
        ## Choose a random start state
        r, c = cg.env.random_start_state()
        g = cg.env.get_goal_state()
        trial_data = Trial((r,c), g, cg.current_reward_estimate(), cg.env.get_world())
        steps = 0
        while cg.env.world[r,c] != g and steps < max_steps:
            ## Agent Plans an action given the current state, based on current policy
            action, local_policy = cg.chooseAction(r, c)

            ## Tell the human what action the agent plans to take from the current state:
            print(f"Agent is currently at ({r},{c}) on {cg.env.world[r,c]}")
            cg.env.visualize_environment(r,c)
            cg.env.inform_human(action)

            ## Get feedback for the planned action before actually taking it.
            feedback_str = cg.env.acquire_feedback(action, r, c, source_is_human=True)
            feedback = None
            ## Classify the feedback
            feedback_type = cg.env.classify_feedback(feedback_str)
            if feedback_type == cg.env.SCALAR_FEEDBACK:
                print("Scalar Feedback Provided")
                scalar = cg.env.feedback_to_scalar(feedback_str)
                feedback = scalar
                pi, Q, loss, rk = cg.scalar_update(scalar, action, r, c)
                print(f"Updated reward: {rk}")

            elif feedback_type == cg.env.ACTION_FEEDBACK:
                print("Action Feedback Provided")
                #Collect Trajectories
                trajacts, trajcoords = cg.env.feedback_to_demonstration(feedback_str, r, c)
                feedback = trajacts[0]
                # trajacts, trajcoords = cg.env.acquire_human_demonstration(max_length=1)
                ## Update the reward function based on the trajectory
                pi, Q, loss, rk = cg.action_update(trajacts, trajcoords)
                print(f"Updated reward: {rk}")

            else:
                print("Invalid feedback provided. Ignoring.")

            trial_data.add_transition(
                steps, cg.current_reward_estimate(),
                (r,c), action, local_policy, feedback_type, feedback
            )

            ## Perform actual transition
            r,c = cg.env.step(r,c, action)
            steps += 1

        if steps == max_steps:
            print("Agent unable to learn after {steps} steps, going to next trial.")
        all_trial_data.append(trial_data)

        tensor_time.append(copy.deepcopy(rk))
        if ep % 1 == 0:
            for x in tensor_time:
                print(x)
            print('Plotting Policy:')
            cg.plotpolicy(pi)

    ## Save the data for this experiment. The environment is small enough
    ## that we can just regenerate the policy at each time step from the reward function.
    ## Note that the reward function is the reward AFTER feedback is obtained
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"exp_{ts}.pkl","wb") as f:
        pickle.dump(all_trial_data, f)

    return all_trial_data

def compute_violations(data, cg):
    """
    Computes the violations at the end of each episode
    Computes the violations during training of each timestep during each episode
    """
    nrows, ncols = cg.env.world.shape
    acts = cg.env.actions
    grid_map = cg.env.world

    def policyViolations(pi, do_print=True):
        violation_map = np.zeros((nrows, ncols))
        iteration_map = np.zeros((nrows, ncols))
        for i in range(nrows):
            for j in range(ncols):
                grid = []
                for r in range(nrows):
                    line = []
                    for c in range(ncols):
                        line += [6]
                    grid += [line]
                it, viol = stateViolations(grid, pi, i, j)
                violation_map[i][j] = viol
                iteration_map[i][j] = it
        if do_print:
            print("Policy Violation Map:")
            print(violation_map)
            print("Iteration Map:")
            print(iteration_map)
            print("Average Policy Violation Count: " + str(np.mean(violation_map)))
            # print("Standard Deviation Violation Count: " + str(round(np.std(violation_map), 3)))
            print("Average Iteration Count: " + str(np.mean(iteration_map)))
            # print("Standard Deviation Iteration Count: " + str(round(np.std(iteration_map), 3)))
        return (np.mean(iteration_map), np.mean(violation_map))
        # returns number of violations in a state

    def stateViolations(grid, pi, r, c):
        if grid[r][c] != 6: return (0, 0)
        maxprob = max(pi[r*ncols+c, :])
        a = 6
        for ana in range(5):
            if pi[r*ncols+c, ana] == maxprob: a = ana
        grid[r][c] = a
        r += acts[a][0]
        c += acts[a][1]
        it, viol = stateViolations(grid, pi, r, c)
        if grid[r][c] < 4:
            it += 1
        tile_type = grid_map[r][c]
        if tile_type == 1 or tile_type == 2 or tile_type == 3:
            viol += 1
        if tile_type == 0 and a == 4:
            viol += 1 ## Violation by staying in the 0 zone
        return (it, viol)

    violations_per_trial = []
    for trial in data:
        violations = []
        r = trial.reward_estimate
        print(f'Reawrd Estimate: {r}')
        cg.env.world = trial.grid_map
        cg.set_reward_estimate(r)
        pi, Q = cg.forward()
        violations.append(policyViolations(pi))
        for transition in trial.transitions:
            r = transition.reward_estimate
            print(f'Reawrd Estimate: {r}')
            cg.set_reward_estimate(r)
            pi, Q = cg.forward()
            violations.append(policyViolations(pi))
        violations_per_trial.append(violations)

    return violations_per_trial

def plot_violations(violations_list, save_prefix, show=True):
    """
    Plots violations as a function of trial
    Plots violations over the course of a trial
    """
    if not show:
        plt.ioff()

    ## Plot policy violations afte reach trial has been completed
    trial_violations = []
    for trial in violations_list:
        trial_violations.append(trial[-1][1])
    trials = np.arange(len(violations_list))+1

    plt.plot(trials, trial_violations)
    plt.xlabel('Trial')
    plt.ylabel('Policy Violation Average After Trial')
    plt.title('Average Violation as a function of Trials')
    if show:
        plt.show()
    else:
        plt.savefig(save_prefix+".per_trial.pdf", bbox_inches='tight')
        plt.close()


    ## Plot of evolution of policy violations as trials progress
    offset = 0
    for trial in violations_list:
        ## Column 1 is violations, Column 0 is iterations
        data = np.array(trial)[:,1]
        length = len(data)
        x = np.arange(length) + offset
        offset += length
        print(f'{data}')
        plt.plot(x, data)
    plt.xlabel('Steps')
    plt.ylabel('Policy Violation Average After Timestep')
    plt.title('Violations Over Time')
    if show:
        plt.show()
    else:
        plt.savefig(save_prefix+".per_step.pdf", bbox_inches='tight')
        plt.close()

def parse_args():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # yapf: disable
    parser.add_argument('--mode', help='"train" or "test" mode', required=True, type=str)
    parser.add_argument('--dataset', help='path to dataset', type=str)
    # yapf: enable

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    all_training_data = None
    ## Select a world
    idx = 0
    grid_maps, state_starts, viz_starts = Worlds.define_worlds()
    env = Environment(grid_maps[idx], state_starts[idx], viz_starts[idx], Worlds.categories)
    cg = ComputationGraph(env)
    if args.mode == "train":
        episodes = 10

        all_training_data = train(episodes, cg)
    elif args.mode == "test":
        if args.dataset is None:
            print(f"Dataset path must be specified.")
            exit()

        if not os.path.isfile(args.dataset):
            print(f"Dataset path is not a file.")
            exit()

        with open(args.dataset, 'rb') as f:
            all_training_data = pickle.load(f)
    else:
        print(f"Mode must be train or test")
        exit()

    plt.rcParams.update({'font.size': 20})
    all_violations = compute_violations(all_training_data, cg)
    plot_violations(all_violations, args.dataset, show=False)
