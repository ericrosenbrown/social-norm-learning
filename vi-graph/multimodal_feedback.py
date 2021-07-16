import random
from itertools import accumulate
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
    class lossdict(dict):
        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)

        def set_decode(self, mapping):
            self.action_mapping = mapping

        def __str__(self):
            parts = []
            for k, v in self.items():
                parts.append('{0}: {1} -> {2}'.format(k[0], self.action_mapping[k[1]], str(v[-1])))

            return '\n'.join(parts)

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

        ## Losses are recorded state by state for actions
        self.recorded_losses = ComputationGraph.lossdict()
        self.recorded_losses.set_decode({0: 'U', 1: 'R', 2: 'D', 3: 'L', 4: 'S'})

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

    def compute_expert_loss(self, pi, trajacts, trajcoords):
        nrows, ncols = self.env.world.shape
        loss = 0

        ## Update the losses for all the demonstrations that have been seen so far
        for sa in self.recorded_losses:
            state, acti = sa
            example_loss = -torch.log(pi[state[0]*ncols+state[1]][acti])
            self.recorded_losses[sa].append(example_loss.cpu().detach().numpy())

        ## Update the new loss if applicable
        for i in range(len(trajacts)):
            acti = trajacts[i]
            state = trajcoords[i]
            loss += -torch.log(pi[state[0]*ncols+state[1]][acti])

            if (state, acti) not in self.recorded_losses:
                self.recorded_losses[(state,acti)] = [loss.cpu().detach().numpy()]
        return loss

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

class TrialBuffer:
    def __init__(self):
        self.all_trials = []
        self.feedback_statistics = {}

    def register_trial(self, trial):
        self.all_trials.append(trial)
        for k,v in trial.feedback_indices.items():
            if k not in self.feedback_statistics:
                self.feedback_statistics[k] = []

    def update_statistics(self):
        trial = self.all_trials[-1]
        for feedback_type, indices in trial.feedback_indices.items():
            self.feedback_statistics[feedback_type].append(len(indices))

    def sample_feedback_type(self, feedback_type, batch_size):
        ## How many samples are in our buffer?
        total_current = len(self.all_trials[-1].feedback_indices[feedback_type])
        total_feedback = [x for x in self.feedback_statistics[feedback_type]]
        total_feedback.append(total_current)
        total = sum(total_feedback)
        selection = None
        if batch_size < total:
            selection = random.sample(range(total), batch_size)
        else:
            selection = [x for x in range(total)]

        ## Sort in the indices for iteration
        selection.sort()
        transitions = [None for x in range(len(selection))]
        trial_idx = 0
        selection_idx = 0
        prev_accum = 0
        for x in accumulate(total_feedback):
            while selection_idx < len(selection) and selection[selection_idx] < x:
                idx = selection[selection_idx] - prev_accum
                try:
                    data_idx = self.all_trials[trial_idx].feedback_indices[feedback_type][idx]
                except IndexError:
                    print(f'Selection: {selection}')
                    print(f'x: {x}, selection[selection_idx]={selection[selection_idx]}')
                    print(f'prev_accum: {prev_accum}, idx={idx}')
                    print(f'trial_idx: {trial_idx}')
                    print(f'total_feedback: {total_feedback}')
                transitions[selection_idx] = self.all_trials[trial_idx].transitions[data_idx]
                selection_idx += 1
            prev_accum = x
            trial_idx += 1
        return transitions

class Trial:
    def __init__(self, initial_state, goal, reward_estimate, grid_map):
        self.initial_state = initial_state
        self.goal = goal
        self.reward_estimate = reward_estimate
        self.grid_map = grid_map
        self.transitions = []
        self.feedback_indices = dict()

    def register_feedback_type(self, feedback_type):
        self.feedback_indices[feedback_type] = []

    def add_transition(self, t, r, s, a, p, ft, fv):
        self.transitions.append(Transition(t, r, s, a, p, ft, fv))
        if ft in self.feedback_indices:
            idx = len(self.transitions) - 1
            self.feedback_indices[ft].append(idx)

    def update_last_transition_reward(self, r):
        last_transition = self.transitions[-1]
        self.transitions[-1] = Transition(*((last_transition[0], r) + last_transition[2:]))

def train(episodes, cg, prepop_trial_data=None):
    """
    Train the agent's estimate of the reward function
    """
    tensor_time = []
    pi, Q, loss, rk = None, None, None, None
    max_steps = 100
    trial_buffer = TrialBuffer()
    ## Prepopulate the TrialBuffer with feedback examples
    if prepop_trial_data is not None:
        trial_buffer.register_trial(prepop_trial_data)
        trial_buffer.update_statistics()
        
    for ep in range(episodes):
        ## Initialize episode:
        ## Choose a random start state
        r, c = cg.env.random_start_state()
        g = cg.env.get_goal_state()

        trial_data = Trial((r,c), g, cg.current_reward_estimate(), cg.env.get_world())
        trial_data.register_feedback_type(cg.env.SCALAR_FEEDBACK)
        trial_data.register_feedback_type(cg.env.ACTION_FEEDBACK)
        trial_buffer.register_trial(trial_data)

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
            else:
                print("Invalid feedback provided. Ignoring.")

            trial_data.add_transition(
                steps, cg.current_reward_estimate(),
                (r,c), action, local_policy, feedback_type, feedback
            )

            if feedback_type == cg.env.ACTION_FEEDBACK:
                ## Update the reward function based on a batch of expert transitions from
                ## the feedback buffer
                batch_transitions = trial_buffer.sample_feedback_type(cg.env.ACTION_FEEDBACK, 100)
                ## Convert transitions to trajacts, trajcoords
                trajacts = [tr.feedback_value for tr in batch_transitions]
                trajcoords = [tr.state for tr in batch_transitions]
                pi, Q, loss, rk = cg.action_update(trajacts, trajcoords)

                ## Because the reward in the transition should be from AFTER feedback
                trial_data.update_last_transition_reward(cg.current_reward_estimate())

                cg.compute_expert_loss(pi, [feedback], [(r,c)])
                print(f"Updated reward: {rk}")

            print(cg.recorded_losses)

            ## Perform actual transition
            r,c = cg.env.step(r,c, action)
            steps += 1

        if steps == max_steps:
            print(f"Agent unable to learn after {steps} steps, going to next trial.")
        trial_buffer.update_statistics()

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
        pickle.dump(trial_buffer.all_trials, f)
        pickle.dump(cg.recorded_losses, f)

    return trial_buffer.all_trials, cg.recorded_losses

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
        return iteration_map, violation_map
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
    detailed_violations_per_trial = []
    for trial in data:
        violations = []
        r = trial.reward_estimate
        print(f'Reawrd Estimate: {r}')
        cg.env.world = trial.grid_map
        cg.set_reward_estimate(r)
        pi, Q = cg.forward()
        iteration_map, violation_map = policyViolations(pi)
        violations.append((np.mean(iteration_map),np.mean(violation_map)))
        detailed_violations_per_trial.append((iteration_map, violation_map))
        for transition in trial.transitions:
            r = transition.reward_estimate
            print(f'Reawrd Estimate: {r}')
            cg.set_reward_estimate(r)
            pi, Q = cg.forward()
            iteration_map, violation_map = policyViolations(pi)
            violations.append((np.mean(iteration_map),np.mean(violation_map)))
            detailed_violations_per_trial.append((iteration_map, violation_map))
        violations_per_trial.append(violations)

    return violations_per_trial, detailed_violations_per_trial

def plot_violations(violations_list, detailed_violations, save_prefix, show=True):
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

    ## Plots of detailed policy violations as trials progress,
    ## The violation map is flattened to 1D and shown as a time series
    ## First, create the flattened timeseries matrix of violations
    total_steps = len(detailed_violations)
    det_vio = [x[1] for x in detailed_violations]
    flattened_violations = np.array(det_vio).reshape((total_steps, 50)).T
    plt.matshow(flattened_violations)
    plt.yticks(np.arange(50))
    plt.grid(True, which='both', axis='y', color='r')
    plt.xlabel('Steps')
    plt.ylabel('State')
    plt.title('States in which Policy Violations Occured at Timestep')
    if show:
        plt.show()
    else:
        plt.savefig(save_prefix+".detailed_per_step_violations.pdf", bbox_inches='tight')
        plt.close()

def compute_demonstration_losses(data, cg, demonstration_losses):
    ## Recompute the demonstration losses from scratch for now (otherwise, could
    ## just backfill based on the length, but that seems a little complicated.
    nrows, ncols = cg.env.world.shape
    acts = cg.env.actions
    grid_map = cg.env.world

    def update_losses(pi, demo):
        for k in demo:
            state, acti = k
            loss = -torch.log(pi[state[0]*ncols+state[1]][acti])
            demo[k].append(loss.cpu().detach().numpy())
        return demo
    
    demo_losses = {k: [] for k in demonstration_losses}
    for trial in data:
        r = trial.reward_estimate
        print(f'Reward Estimate: {r}')
        cg.env.world = trial.grid_map
        cg.set_reward_estimate(r)
        pi, Q = cg.forward()
        demo_losses = update_losses(pi, demo_losses)

        for transition in trial.transitions:
            r = transition.reward_estimate
            print(f'Reawrd Estimate: {r}')
            cg.set_reward_estimate(r)
            pi, Q = cg.forward()
            demo_losses = update_losses(pi, demo_losses)

    ## Confirm that all lists have the same length:
    len_check = None
    all_same_length = True
    for k, v in demo_losses.items():
        if len_check is None:
            len_check = len(v)
        else:
            all_same_length = all_same_length & (len_check == len(v))
        print(k, len(v))
        assert(all_same_length)

    return demo_losses

def plot_losses(demonstration_losses, show=True):
    """
    Plots demo loss as a function of timesteps
    """
    if not show:
        plt.ioff()

    avg_loss = None
    N = len(demonstration_losses)
    for k, v in demonstration_losses.items():
        if avg_loss is not None:
            avg_loss[:] = avg_loss[:] + np.array(v)
        else:
            avg_loss = np.array(v)
        x = np.arange(len(v))
        plt.plot(x, v, label=str(k))
    avg_loss[:] = avg_loss[:] / N
    x = np.arange(len(avg_loss))
    plt.plot(x, avg_loss, label='avg', linewidth=1.5)

    #for k, v in demonstration_losses.items():
    #    x = np.arange(len(v))
    #    plt.plot(x[-1], v[-1], str(k))
    #x = np.arange(len(avg_loss))
    #plt.plot(x[-1], avg_loss[-1], 'avg')

    plt.xlabel('Steps')
    plt.ylabel('Loss After Timestep')
    plt.title('Losses of Demostrated Examples')
    #plt.legend()
    if show:
        plt.show()
    else:
        plt.savefig("loss.per_step.pdf", bbox_inches='tight')
        plt.close()
    

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', help='"train" or "test" mode', required=True, type=str)
    parser.add_argument('--dataset', help='path to dataset', type=str)
    parser.add_argument('--prepopulate', help='prepopulate with action feedback',
        dest='prepopulate', action='store_true')
    parser.add_argument('--hide', help='hide plots, used for autosaving plots',
        dest='hide', action='store_true')
    return parser.parse_args()

def prepopulate(cg):
    r, c = cg.env.random_start_state()
    g = cg.env.get_goal_state()
    prepop_trial_data = Trial((r,c), g, cg.current_reward_estimate(), cg.env.get_world())
    prepop_trial_data.register_feedback_type(cg.env.SCALAR_FEEDBACK)
    prepop_trial_data.register_feedback_type(cg.env.ACTION_FEEDBACK)

    nrows, ncols = cg.env.world.shape
    acts = cg.env.actions
    grid_map = cg.env.world

    steps = 0
    for r in range(nrows):
        for c in range(ncols):
            action, local_policy = cg.chooseAction(r, c)
            feedback_str = cg.env.acquire_feedback(action, r, c, source_is_human=False)
            feedback = None
            ## Classify the feedback
            feedback_type = cg.env.classify_feedback(feedback_str)
            if feedback_type == cg.env.ACTION_FEEDBACK:
                print("Action Feedback Provided")
                #Collect Trajectories
                trajacts, trajcoords = cg.env.feedback_to_demonstration(feedback_str, r, c)
                feedback = trajacts[0]
            prepop_trial_data.add_transition(
                steps, cg.current_reward_estimate(),
                (r,c), action, local_policy, feedback_type, feedback
            )
            steps += 1

    return prepop_trial_data

if __name__ == '__main__':
    args = parse_args()
    all_training_data = None
    demonstration_losses = None
    ## Select a world
    idx = 0
    grid_maps, state_starts, viz_starts = Worlds.define_worlds()
    env = Environment(grid_maps[idx], state_starts[idx], viz_starts[idx], Worlds.categories)
    cg = ComputationGraph(env)
    prepop_trial_data = None
    show_plots = True
    if args.hide:
        show_plots = False
        if args.dataset is None:
            print(f"Save prefix must be specified if hiding plots.")
            exit()

    if args.mode == "train":
        if args.prepopulate:
            print("Prepopulating trial data")
            prepop_trial_data = prepopulate(cg)
        episodes = 10
        all_training_data, demonstration_losses = train(episodes, cg, prepop_trial_data)
    elif args.mode == "test":
        if args.dataset is None:
            print(f"Dataset path must be specified.")
            exit()

        if not os.path.isfile(args.dataset):
            print(f"Dataset path is not a file.")
            exit()

        with open(args.dataset, 'rb') as f:
            all_training_data = pickle.load(f)
            ## Check if we have the loss saved in the pickle file
            try:
                demonstration_losses = pickle.load(f)
            except EOFError:
                print("No demonstration losses in this dataset")
    else:
        print(f"Mode must be train or test")
        exit()

    plt.rcParams.update({'font.size': 20})
    all_violations, detailed_violations = compute_violations(all_training_data, cg)
    plot_violations(all_violations, detailed_violations, args.dataset, show=show_plots)
    if demonstration_losses is not None:
        demo_losses = compute_demonstration_losses(all_training_data, cg, demonstration_losses)
        plot_losses(demo_losses, show=show_plots)
