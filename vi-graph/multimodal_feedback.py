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

    def request_feedback(self, cur_r, cur_c, force=True):
        """
        Uses heuristics to determine whether or not to request feedback
        from the human.

        Current heuristics: number of violations, goal success rate,
        "have I recieved feedback for this state?"

        Stochastic Policy values
        """
        if force:
            return force

        nrows, ncols = self.env.world.shape
        acts = self.env.actions
        grid_map = self.env.world

        def plan(rs, cs, pi, can_reach, steps, max_steps):
            r, c = rs, cs
            success = 0.0
            violations = 0
            while steps < max_steps:
                if grid_map[r][c] == 4 or can_reach[r*ncols + c] > 0.5:
                    success = 1.0
                    break
                maxprob = max(pi[r*ncols+c, :])
                a = 6
                for ana in range(5):
                    if pi[r*ncols+c, ana] == maxprob: a = ana
                r += acts[a][0]
                c += acts[a][1]
                steps += 1
                if grid_map[r][c] in (1,2,3):
                    violations += 1
            return success, violations

        def trajSuccess(rs, cs, pi):
            """
            Computes goal success of a trajectory
            """
            success = np.zeros(nrows*ncols)
            goal, violations = plan(rs, cs, pi, success, 0, nrows*ncols)
            return goal, violations

        def goalSuccess(pi):
            """
            For every possible state, does planning with
            the current policy lead to the goal state?
            """
            success = np.zeros(nrows*ncols)
            for i in range(nrows):
                for j in range(ncols):
                    success[i*ncols + j], _ = plan(i, j, pi, success, 0, nrows*ncols)
            return np.mean(success)

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

        ## Compute Heuristics for making the decisions
        pi, Q = cg.forward()
        success_rate = goalSuccess(pi)
        iters, viols = policyViolations(pi)
        #success_rate, viols = trajSuccess(cur_r, cur_c, pi)

        if success_rate < 1.0 or np.mean(viols) > 0.02:
            return True
        else:
            return False

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

    def amount_of(self, feedback_type):
        if feedback_type not in self.feedback_indices:
            return 0
        else:
            return len(self.feedback_indices[feedback_type])

    def get_effort(self, feedback_types):
        return sum(len(self.feedback_indices[k]) for k in feedback_types)

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

class GridWorldRewardModel:
    def __init__(self, reward_features, env, gamma, trans_dict, trans_tuple):
        self.gamma = gamma
        self.env = env
        self.matmap = env.get_matmap()
        nrows, ncols, ncategories = self.matmap.shape
        self.nrows = nrows
        self.ncols = ncols
        self.ncats = ncategories
        ## Grid map is a 2D representation of the GridWorld. Each element is the category
        self.grid_map = torch.tensor(env.world, dtype=int, requires_grad=False)
        ## Observation and action space is given by the flattened grid_world, it is 1D.
        self.obs_space = torch.arange(nrows*ncols, dtype=int)
        self.act_space = torch.arange(len(env.actions), dtype=int)
        ## Represents possible actions
        self.actions = torch.arange(len(env.actions), dtype=int, requires_grad=False)
        ## R(s,a,s') = R(s,a,\phi(s')), feature based rewards vector
        self.trans_dict = trans_dict
        self.trans_tuple = trans_tuple

        self.feature_rewards = torch.tensor(reward_features, dtype=env.dtype, requires_grad=True)
        self._forward()

    def _forward(self):
        new_rk = self.feature_rewards.unsqueeze(0)
        new_rk = new_rk.unsqueeze(0)
        new_rk = new_rk.expand(self.nrows, self.ncols, self.ncats)
        ## Dot product to obtain the reward function applied to the matrix map
        rfk = torch.mul(self.matmap, new_rk)
        rfk = rfk.sum(axis=-1) ## 2D representation, i.e. recieve R((r,c)) reward for arriving at (r,c)
        self.reward_model = rfk.view(self.nrows*self.ncols) ## flattened 1D view of the 2D grid
        ## Create 3D verasion of rewrd model: (s,a,s'). The above version corresponds with s'
        ## R(s,a,s')
        self.full_reward_model = torch.zeros((self.nrows*self.ncols, len(env.actions), self.nrows*self.ncols))
        for s,a,sp in self.trans_tuple:
            self.full_reward_model[s,a,sp] = self.reward_model[sp]
        self.canonicalized_reward = self.get_canonicalized_reward(self.trans_dict, self.trans_tuple)

    def update(self, reward_features):
        self.feature_rewards = torch.tensor(reward_features, dtype=env.dtype, requires_grad=True)
        self._forward()

    def expected_reward_from_s(self, s, transitions):
        """
        Computes the mean reward exactly starting in state s, averaged
        over the possible (a,s') allowed from state s
        (s,a,s') are in terms of 1D representation, i.e. s, s' \in {0,...,|S|-1}

        transitions is a {s: {a: s'}} dict
        """
        return torch.mean(self.reward_model[[sp for a, sp in transitions[s].items()]])

    def expected_reward_over_sas(self, transitions):
        """
        transitions is a tuple( tuple(s,a,s'), ... )
        """
        ## NOTE: We need to make a 1D tensor that gathers from specific indices represented by sasp
        ## This can be done with the 1D reward model
        return torch.mean(self.reward_model[[sasp[2] for sasp in transitions]])

    def get_canonicalized_reward(self, transitions_dict, transitions_tuple):
        canonicalized = torch.clone(self.full_reward_model) ## R(s,a,s')
        ## Below, used to compute R(s',A,S') and R(s,A,S')
        mean_from_state = torch.tensor(
            [self.expected_reward_from_s(state, transitions_dict) for state in range(self.env.world.size)]
        )

        ## Compute E[R(S,A,S')]
        mean_reward = torch.sum(self.full_reward_model)/len(transitions_tuple)
        for s,a,sp in transitions_tuple:
            canonicalized[s,a,sp] += (self.gamma*mean_from_state[sp] - mean_from_state[s] - self.gamma*mean_reward)
        return canonicalized

    def epic_distance(self, other, samples):
        """
        sample is (s,a,s') tuples
        """
        shape = self.canonicalized_reward.shape
        S = shape[0]
        A = shape[1]
        ra = torch.flatten(self.canonicalized_reward)
        rb = torch.flatten(other.canonicalized_reward)

        idx = (lambda s,a,sp: s*A*S + a*S + sp)
        ## Sample indices
        indices = np.array([idx(s,a,sp) for s,a,sp in samples])
        return self.pearson_distance(ra[indices], rb[indices])

    def pearson_distance(self, ra, rb):
        mu_a = torch.mean(ra)
        mu_b = torch.mean(rb)
        var_a = torch.mean(torch.square(ra - mu_a))
        var_b = torch.mean(torch.square(rb - mu_b))

        cov = torch.mean((ra - mu_a) * (rb - mu_b))
        corr = cov / torch.sqrt(var_a * var_b)
        corr = torch.clamp(corr, -1.0, 1.0)
        return torch.sqrt((1.0-corr)/2.0)

def train(episodes, cg, prepop_trial_data=None, force_feedback=True):
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
        r, c = next(cg.env.random_start_state())
        g = cg.env.get_goal_state()

        trial_data = Trial((r,c), g, cg.current_reward_estimate(), cg.env.get_world())
        trial_data.register_feedback_type(cg.env.SCALAR_FEEDBACK)
        trial_data.register_feedback_type(cg.env.ACTION_FEEDBACK)
        trial_data.register_feedback_type(cg.env.NO_FEEDBACK)
        trial_buffer.register_trial(trial_data)

        steps = 0
        while cg.env.world[r,c] != g and steps < max_steps:
            ## Agent Plans an action given the current state, based on current policy
            action, local_policy = cg.chooseAction(r, c)

            if cg.request_feedback(r, c, force=force_feedback):

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
            else:
                trial_data.add_transition(
                    steps, cg.current_reward_estimate(),
                    (r,c), action, local_policy, cg.env.NO_FEEDBACK, None
                )

            ## endif cg.request_feedback()

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

def compute_effort(data, cg):
    ## Effort = How much feedback was given
    effort = [trial.get_effort((cg.env.ACTION_FEEDBACK, cg.env.SCALAR_FEEDBACK)) for trial in data]
    ## Distribution: Quantity of Action, Quantity of Scalar
    dist = {
        cg.env.ACTION_FEEDBACK: None,
        cg.env.SCALAR_FEEDBACK: None,
        cg.env.NO_FEEDBACK: None
    }
    for k in dist:
        dist[k] = [trial.amount_of(k) for trial in data]
    return effort, dist

def compute_goal_success(data, cg):
    """
    Computes the goal success at each timestep
    Computes the goal success at the end of the episode (= goal success of last timestep)
    """
    nrows, ncols = cg.env.world.shape
    acts = cg.env.actions
    grid_map = cg.env.world

    def plan(rs, cs, pi, can_reach, steps, max_steps):
        r, c = rs, cs
        success = 0
        while steps < max_steps:
            if grid_map[r][c] == 4 or can_reach[r*ncols + c] > 0.5:
                success = 1
                break
            maxprob = max(pi[r*ncols+c, :])
            a = 6
            for ana in range(5):
                if pi[r*ncols+c, ana] == maxprob: a = ana
            r += acts[a][0]
            c += acts[a][1]
            steps += 1
        return success

    def goalSuccess(pi):
        """
        For every possible state, does planning with
        the current policy lead to the goal state?
        """
        success = np.zeros(nrows*ncols)
        for i in range(nrows):
            for j in range(ncols):
                success[i*ncols + j] = plan(i, j, pi, success, 0, nrows*ncols)
        return np.mean(success)

    goal_rate_per_trial = []
    for trial in data:
        goal_rate = []
        r = trial.reward_estimate
        print(f'Reawrd Estimate: {r}')
        cg.env.world = trial.grid_map
        cg.set_reward_estimate(r)
        pi, Q = cg.forward()
        success_rate = goalSuccess(pi)
        goal_rate.append(success_rate)
        for transition in trial.transitions:
            r = transition.reward_estimate
            print(f'Reawrd Estimate: {r}')
            cg.set_reward_estimate(r)
            pi, Q = cg.forward()
            success_rate = goalSuccess(pi)
            goal_rate.append(success_rate)
        goal_rate_per_trial.append(goal_rate)
    return goal_rate_per_trial

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

def min_mean_max_violations(violations):
    """
    Obtain the min, mean, and max of the input violations
    """
    min_violations = []
    max_violations = []
    avg_violations = []
    end_violations = []
    ini_violations = []
    len_trial = []
    for trial_violations in violations:
        min_violations.append(min(v[1] for v in trial_violations))
        max_violations.append(max(v[1] for v in trial_violations))
        end_violations.append(trial_violations[-1][1])
        ini_violations.append(trial_violations[0][1])
        steps = len(trial_violations)
        avg_violations.append(sum(v[1] for v in  trial_violations) / steps)
        len_trial.append(steps)

    return {
        'min': min_violations,
        'max': max_violations,
        'avg': avg_violations,
        'end': end_violations,
        'ini': ini_violations,
        'len': len_trial
    }

def min_mean_max_goal_success(rates):
    """
    Obtain the min, mean, and max of the input goal success rates
    """
    min_success = []
    max_success = []
    avg_success = []
    end_success = []
    ini_success = []
    len_trial = []
    for trial_rates in rates:
        min_success.append(min(r for r in trial_rates))
        max_success.append(max(r for r in trial_rates))
        end_success.append(trial_rates[-1])
        ini_success.append(trial_rates[0])
        steps = len(trial_rates)
        avg_success.append(sum(r for r in  trial_rates) / steps)
        len_trial.append(steps)

    return {
        'min': min_success,
        'max': max_success,
        'avg': avg_success,
        'end': end_success,
        'ini': ini_success,
        'len': len_trial
    }

def plot_group_violations(
    group_violations, group_success, group_effort,
    group_act, group_sca, group_non,
    group_pra, group_prs, group_prn, show=True):
    """
    Plots violations as a function of trial. Specifically,
    plots the max, mean, and min of a trail, averaged over a set of seeds.

    The structure of group_violations is:
    {
        grp_name: {
            'min': [average of seeds of ep1 val, ...],
            'max': [average of seeds of ep1 val, ...],
            'avg': [average of seeds of ep1 val, ...],
            'end': [average of seeds of ep1 val, ...],
        }
    }

    """
    styles = {
        'min': ':',
        'max': '--',
        'avg': '-.',
        'end': '-'
    }
    #color_list = ['blue','green', 'red', 'cyan', 'magenta', 'yellow', 'black']
    color_list = ['magenta','gold', 'green', 'blue', 'red', 'cyan', 'black']
    colors = {k:v for k,v in zip(group_violations, color_list)}

    print(group_violations)

    ## Violation Averages
    for grp, avgs in group_violations.items():
        x = [n+1 for n in range(len(avgs['end']))]
        for k in ('min', 'max', 'end'):
            plt.plot(x, avgs[k], linestyle=styles[k], color=colors[grp], label=grp+'_'+k)
        plt.fill_between(x, avgs['min'], avgs['max'], color=colors[grp], alpha=0.2)
    plt.xlabel('Trial')
    plt.ylabel('Violations')
    plt.title('Violation Averages')
    if show:
        plt.legend()
        plt.show()
    else:
        plt.savefig("violation_averages.per_trial.pdf", bbox_inches='tight')
        plt.close()

    ## Goal Success Averages
    for grp, avgs in group_success.items():
        x = [n+1 for n in range(len(avgs['end']))]
        for k in ('min', 'max', 'end'):
            plt.plot(x, avgs[k], linestyle=styles[k], color=colors[grp], label=grp+'_'+k)
        plt.fill_between(x, avgs['min'], avgs['max'], color=colors[grp], alpha=0.2)
    plt.xlabel('Trial')
    plt.ylabel('Goal Success')
    plt.title('Average Goal Success Rate')
    if show:
        plt.legend()
        plt.show()
    else:
        plt.savefig("goal_success.per_trial.pdf", bbox_inches='tight')
        plt.close()

    ## Cumulative Effort
    for grp, avgs in group_effort.items():
        x = [n+1 for n in range(len(avgs['min']))]
        for k in ('min', 'avg', 'max'):
            plt.plot(x, np.cumsum(avgs[k]), linestyle=styles[k], color=colors[grp], label=grp+'_effort')
        plt.fill_between(x, np.cumsum(avgs['min']), np.cumsum(avgs['max']), color=colors[grp], alpha=0.2)
    plt.xlabel('Trial')
    plt.ylabel('Effort')
    plt.title('Cumulative Effort')
    if show:
        plt.legend()
        plt.show()
    else:
        plt.savefig("cumulative_effort.per_trial.pdf", bbox_inches='tight')
        plt.close()

    ## Length per Trial
    for grp, avgs in group_violations.items():
        x = [n+1 for n in range(len(avgs['len']))]
        for k in ('min', 'avg', 'max'):
            plt.plot(x, avgs['len'], color=colors[grp], label=grp+'_len')
        
    ## Effort per Trial
    for grp, avgs in group_effort.items():
        x = [n+1 for n in range(len(avgs['min']))]
        for k in ('min', 'avg', 'max'):
            plt.plot(x, avgs[k], linestyle=styles[k], color=colors[grp], label=grp+'_effort')
        plt.fill_between(x, avgs['min'], avgs['max'], color=colors[grp], alpha=0.2)
    plt.xlabel('Trial')
    plt.ylabel('Effort')
    plt.title('Effort')
    if show:
        plt.legend()
        plt.show()
    else:
        plt.savefig("effort.per_trial.pdf", bbox_inches='tight')
        plt.close()
    
    ## Feedback Types per Trial
    for grp, avgs in group_act.items():
        x = [n+1 for n in range(len(avgs['min']))]
        for k in ('min', 'avg', 'max'):
            plt.plot(x, avgs[k], linestyle=styles[k], color=colors[grp], label=grp+'_effort')
        plt.fill_between(x, avgs['min'], avgs['max'], color=colors[grp], alpha=0.2)
    plt.xlabel('Trial')
    plt.ylabel('Num Action')
    plt.title('Quantity Action Feedback')
    if show:
        plt.legend()
        plt.show()
    else:
        plt.savefig("actions.per_trial.pdf", bbox_inches='tight')
        plt.close()
    ## Feedback Types per Trial
    for grp, avgs in group_pra.items():
        x = [n+1 for n in range(len(avgs['min']))]
        for k in ('min', 'avg', 'max'):
            plt.plot(x, avgs[k], linestyle=styles[k], color=colors[grp], label=grp+'_effort')
        plt.fill_between(x, avgs['min'], avgs['max'], color=colors[grp], alpha=0.2)
    plt.xlabel('Trial')
    plt.ylabel('Pecent Action')
    plt.title('Percent Action Feedback')
    if show:
        plt.legend()
        plt.show()
    else:
        plt.savefig("percent_actions.per_trial.pdf", bbox_inches='tight')
        plt.close()

    ## Feedback Types per Trial
    for grp, avgs in group_sca.items():
        x = [n+1 for n in range(len(avgs['min']))]
        for k in ('min', 'avg', 'max'):
            plt.plot(x, avgs[k], linestyle=styles[k], color=colors[grp], label=grp+'_effort')
        plt.fill_between(x, avgs['min'], avgs['max'], color=colors[grp], alpha=0.2)
    plt.xlabel('Trial')
    plt.ylabel('Num Scalar')
    plt.title('Quantity Scalar Feedback')
    if show:
        plt.legend()
        plt.show()
    else:
        plt.savefig("scalars.per_trial.pdf", bbox_inches='tight')
        plt.close()
    ## Feedback Types per Trial
    for grp, avgs in group_prs.items():
        x = [n+1 for n in range(len(avgs['min']))]
        for k in ('min', 'avg', 'max'):
            plt.plot(x, avgs[k], linestyle=styles[k], color=colors[grp], label=grp+'_effort')
        plt.fill_between(x, avgs['min'], avgs['max'], color=colors[grp], alpha=0.2)
    plt.xlabel('Trial')
    plt.ylabel('Pecent Scalar')
    plt.title('Percent Scalar Feedback')
    if show:
        plt.legend()
        plt.show()
    else:
        plt.savefig("percent_scalars.per_trial.pdf", bbox_inches='tight')
        plt.close()

    ## Feedback Types per Trial
    for grp, avgs in group_non.items():
        x = [n+1 for n in range(len(avgs['min']))]
        for k in ('min', 'avg', 'max'):
            plt.plot(x, avgs[k], linestyle=styles[k], color=colors[grp], label=grp+'_effort')
        plt.fill_between(x, avgs['min'], avgs['max'], color=colors[grp], alpha=0.2)
    plt.xlabel('Trial')
    plt.ylabel('Num None')
    plt.title('Quantity None')
    if show:
        plt.legend()
        plt.show()
    else:
        plt.savefig("no_feedback.per_trial.pdf", bbox_inches='tight')
        plt.close()
    ## Feedback Types per Trial
    for grp, avgs in group_prn.items():
        x = [n+1 for n in range(len(avgs['min']))]
        for k in ('min', 'avg', 'max'):
            plt.plot(x, avgs[k], linestyle=styles[k], color=colors[grp], label=grp+'_effort')
        plt.fill_between(x, avgs['min'], avgs['max'], color=colors[grp], alpha=0.2)
    plt.xlabel('Trial')
    plt.ylabel('Pecent None')
    plt.title('Percent No Feedback')
    if show:
        plt.legend()
        plt.show()
    else:
        plt.savefig("percent_no_feedback.per_trial.pdf", bbox_inches='tight')
        plt.close()

    ## Violations per Total Effort
    for grp, avgs in group_violations.items():
        steps = [0]
        steps.extend(avgs['len'])
        x = np.cumsum(steps)
        for k in ('min', 'max', 'end'):
            y = [avgs['ini'][0]]
            y.extend(avgs[k])
            plt.plot(x, y, linestyle=styles[k], color=colors[grp], label=grp+'_'+k)
        ymin = [avgs['ini'][0]]
        ymin.extend(avgs['min'])
        ymax = [avgs['ini'][0]]
        ymax.extend(avgs['max'])
        plt.fill_between(x, ymin, ymax, color=colors[grp], alpha=0.2)
    plt.xlabel('Total Effort')
    plt.ylabel('Violations')
    plt.title('Violations by Effort')
    if show:
        plt.legend()
        plt.show()
    else:
        plt.savefig("violations.per_cumulative_effort.pdf", bbox_inches='tight')
        plt.close()

    ## Efficiency is Change in Violations between Trials / Amount of Effort for Trial
    ## Violations per Total Effort
    for grp, avgs in group_violations.items():
        steps = [0]
        steps.extend(avgs['len'])
        x = np.cumsum(steps)
        trial_effort = np.array(avgs['len'])
        ymin = None
        ymax = None
        for k in ('min', 'max', 'end'):
            y = [avgs['ini'][0]]
            y.extend(avgs[k])
            L = np.array(y[:-1])
            R = np.array(y[1:])
            efficiency = (L-R)/trial_effort
            if k == 'min':
                ymin = efficiency
            if k == 'max':
                ymax = efficiency
            plt.plot(x[1:], efficiency, linestyle=styles[k], color=colors[grp], label=grp+'_'+k)
        plt.fill_between(x[1:], ymin, ymax, color=colors[grp], alpha=0.2)
    plt.xlabel('Total Effort')
    plt.ylabel('Efficiency')
    plt.title('Improvement Efficiency')
    if show:
        plt.legend()
        plt.show()
    else:
        plt.savefig("efficiency.per_cumulative_effort.pdf", bbox_inches='tight')
        plt.close()

    ## Efficiency per Trial
    for grp, avgs in group_violations.items():
        steps = [0]
        steps.extend(avgs['len'])
        x = np.arange(len(avgs['len'])+1)
        trial_effort = np.array(avgs['len'])
        ymin = None
        ymax = None
        for k in ('min', 'max', 'end'):
            y = [avgs['ini'][0]]
            y.extend(avgs[k])
            L = np.array(y[:-1])
            R = np.array(y[1:])
            efficiency = (L-R)/trial_effort
            if k == 'min':
                ymin = efficiency
            if k == 'max':
                ymax = efficiency
            plt.plot(x[1:], efficiency, linestyle=styles[k], color=colors[grp], label=grp+'_'+k)
        plt.fill_between(x[1:], ymin, ymax, color=colors[grp], alpha=0.2)
    plt.xlabel('Trial')
    plt.ylabel('Efficiency')
    plt.title('Improvement Efficiency')
    if show:
        plt.legend()
        plt.show()
    else:
        plt.savefig("efficiency.per_trial.pdf", bbox_inches='tight')
        plt.close()

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
        plt.savefig(save_prefix+".state_violations_per_step.pdf", bbox_inches='tight')
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

def compute_distances(data, cg):
    ## TODO: Use tabular? Use epic_sample?
    ## Try to implement on own?
    ## Overall steps to compute the EPIC psuedometric:
    ## 0) The reward functions must be R(s,a,s')
    ## 1) Canonicalize the reward functions
    ##    C_Ds_Da(R)(s,a,s') = R(s,a,s')+E[gR(s',A,S') - R(s,A,S') - gR(S,A,S')]
    ## 2) Compute the Pearson distance between the two canonicalized reward functions
    ##    C1 = C(R_1)(S,A,S'), C2 = C(R_2)(S,A,S')
    ##    where C1 and C2 are dependent random variables that depend on S,A,S' which are drawn IID
    ##    from D_S and D_A.
    ## Use EPIC as part of objective function?
    nrows, ncols = cg.env.world.shape
    acts = cg.env.actions
    grid_map = cg.env.world

    trans_dict = cg.env.flattened_sas_transitions()
    trans_tuple = cg.env.all_sas_transitions(trans_dict)
    reward_models = []
    for trial in data:
        r = trial.reward_estimate
        reward_models.append(GridWorldRewardModel(r, cg.env, cg.gamma, trans_dict, trans_tuple))
        for transition in trial.transitions:
            r = transition.reward_estimate
            reward_models.append(GridWorldRewardModel(r, cg.env, cg.gamma, trans_dict, trans_tuple))

    ## Now we have our list of reward models. Canonicalize pairs of consecutive reward models
    n_models = len(reward_models)
    distances = dict()
    distances['epic'] = [] ## EPIC distance
    distances['l2_features'] = [] ## L2 norm of the features
    distances['l2_features_norm'] = [] ## L2 norm of normalized features
    distances['l2_reward'] = [] ## L2 norm of R(s,a,s')
    distances['l2_reward_norm'] = [] ## L2 norm of normalized R(s,a,s')
    distances['l2_canonical'] = [] ## L2 norm of the canonialized reward
    distances['l2_canonical_norm'] = [] ## L2 norm of the normalized canonicalized reward
    A = len(acts)
    S = nrows*ncols
    idx = (lambda s,a,sp: s*A*S + a*S + sp)
    ## Sample indices
    indices = np.array([idx(s,a,sp) for s,a,sp in trans_tuple])
    for idx0, idx1 in zip(range(0,n_models-1),range(1,n_models)):
        cr0 = reward_models[idx0]
        cr1 = reward_models[idx1]
        distances['epic'].append(cr0.epic_distance(cr1, trans_tuple).cpu().detach().numpy())
        distances['l2_features'].append(l2_norm(cr0.feature_rewards, cr1.feature_rewards).cpu().detach().numpy())
        distances['l2_features_norm'].append(
            l2_norm(
                normalize(cr0.feature_rewards),
                normalize(cr1.feature_rewards)
            ).cpu().detach().numpy()
        )
        distances['l2_reward'].append(
            l2_norm(
                torch.flatten(cr0.full_reward_model),
                torch.flatten(cr1.full_reward_model),
                indices=indices
            ).cpu().detach().numpy()
        )
        distances['l2_reward_norm'].append(
            l2_norm(
                normalize(torch.flatten(cr0.full_reward_model), indices=indices),
                normalize(torch.flatten(cr1.full_reward_model), indices=indices)
            ).cpu().detach().numpy()
        )
        distances['l2_canonical'].append(
            l2_norm(
                torch.flatten(cr0.canonicalized_reward),
                torch.flatten(cr1.canonicalized_reward),
                indices=indices
            ).cpu().detach().numpy()
        )
        distances['l2_canonical_norm'].append(
            l2_norm(
                normalize(torch.flatten(cr0.canonicalized_reward), indices=indices),
                normalize(torch.flatten(cr1.canonicalized_reward), indices=indices)
            ).cpu().detach().numpy()
        )
    return distances

def normalize(a, indices=None):
    norm = None
    v = None
    if indices is None:
        v = a
    else:
        v = a[indices]
    norm = torch.norm(v)
    if norm < 1e-24:
        return v
    else:
        return v/norm

def l2_norm(a, b, indices=None):
    """
    Computes the L2 norm ||a-b|| over the indices specified.

    If indices=None, then ||a-b||_2 is computed
    """
    if indices is None:
        return torch.norm(a-b)
    else:
        c = (a-b)[indices]
        return torch.norm(c)

def plot_distances(distances, label, save_prefix, show=True):
    if not show:
        plt.ioff()

    x = np.arange(len(distances))
    plt.plot(x, distances, label=label)

    plt.xlabel('Steps')
    plt.ylabel(label + ' disance between t and t+1')
    plt.title(label + ' distance over time')
    #plt.legend()
    if show:
        plt.show()
    else:
        plt.savefig(save_prefix+"."+label+"_distance.per_step.pdf", bbox_inches='tight')
        plt.close()


def plot_losses(demonstration_losses, save_prefix, show=True):
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
        plt.savefig(save_prefix+".loss.per_step.pdf", bbox_inches='tight')
        plt.close()
    

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', help='"train", "test", or "plot" mode', required=True, type=str)
    parser.add_argument('--dataset', help='path to dataset', type=str)
    parser.add_argument('--prepopulate', help='prepopulate with action feedback',
        dest='prepopulate', action='store_true')
    parser.add_argument('--hide', help='hide plots, used for autosaving plots',
        dest='hide', action='store_true')
    parser.add_argument('--noforce', help='Agent decides whether to request feedback',
        dest='noforce', action='store_true')
    parser.add_argument('--inputs', help='paths to dataset files used for plotting', nargs='+')
    parser.add_argument('--groupings', help='keyword groupings', nargs='+')
    return parser.parse_args()

def prepopulate(cg):
    r, c = next(cg.env.random_start_state())
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
    force_feedback = True
    if args.hide:
        show_plots = False
        if args.dataset is None:
            print(f"Save prefix must be specified if hiding plots.")
            exit()

    if args.noforce:
        force_feedback = False

    if args.mode == "train":
        if args.prepopulate:
            print("Prepopulating trial data")
            prepop_trial_data = prepopulate(cg)
        episodes = 10
        all_training_data, demonstration_losses = train(episodes, cg, prepop_trial_data, force_feedback)
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
    elif args.mode == "plot":
        plt.rcParams.update({'font.size': 20})
        if args.inputs is None or len(args.inputs) == 0:
            print(f'input datasets must be specified')
            exit()
        ## Plot violation averages, maxes, and mins
        group_avg = {grp:None for grp in args.groupings} ## Violations
        group_gsr = {grp:None for grp in args.groupings} ## Goal Success Rate
        group_eff = {grp:None for grp in args.groupings} ## Effort
        group_act = {grp:None for grp in args.groupings} ## Action Effort
        group_sca = {grp:None for grp in args.groupings} ## Scalar Effort
        group_non = {grp:None for grp in args.groupings} ## None   Effort
        group_pra = {grp:None for grp in args.groupings} ## Percent Action
        group_prs = {grp:None for grp in args.groupings} ## Percent Scalar
        group_prn = {grp:None for grp in args.groupings} ## Percent None
        group_num = {grp:0 for grp in args.groupings}

        for dataset in args.inputs:
            if not os.path.isfile(dataset):
                print(f"Dataset path is not a file.")
                exit()
            with open(dataset, 'rb') as f:
                print(f"Loading... {dataset}")
                all_training_data = pickle.load(f)
                ## Check if we have the loss saved in the pickle file
                try:
                    demonstration_losses = pickle.load(f)
                except EOFError:
                    print("No demonstration losses in this dataset")
            ## Fix for missing data
            for t in all_training_data:
                if hasattr(t, "feedback_indices"):
                    if cg.env.NO_FEEDBACK not in t.feedback_indices:
                        t.feedback_indices[cg.env.NO_FEEDBACK] = []
                else:
                    setattr(t, "feedback_indices", dict())
                    t.feedback_indices[cg.env.NO_FEEDBACK] = []
                    t.feedback_indices[cg.env.ACTION_FEEDBACK] = []
                    t.feedback_indices[cg.env.SCALAR_FEEDBACK] = []

            all_violations, detailed_violations = compute_violations(all_training_data, cg)
            violations_dict = min_mean_max_violations(all_violations)
            goal_success_rates = compute_goal_success(all_training_data, cg)
            goal_success_dict = min_mean_max_goal_success(goal_success_rates)
            trial_efforts, effort_dist = compute_effort(all_training_data, cg)
            ## Percentage Action / Scalar
            num_action = np.array(effort_dist[cg.env.ACTION_FEEDBACK])
            num_scalar = np.array(effort_dist[cg.env.SCALAR_FEEDBACK])
            num_none   = np.array(effort_dist[cg.env.NO_FEEDBACK])
            total_num  = num_action + num_scalar + num_none
            per_action = num_action / total_num
            per_scalar = num_scalar / total_num
            per_none   = num_none   / total_num

            ## Determine which group this dataset belongs to
            for grp, grp_avg in group_avg.items():
                if grp in dataset:
                    if grp_avg is None:
                        group_avg[grp] = dict(violations_dict)
                        group_num[grp] = 1
                        group_gsr[grp] = dict(goal_success_dict)
                        group_eff[grp] = {
                            'min': np.array(trial_efforts),
                            'avg': np.array(trial_efforts),
                            'max': np.array(trial_efforts)
                        }
                        group_act[grp] = {
                            'min': np.array(num_action),
                            'avg': np.array(num_action),
                            'max': np.array(num_action)
                        }
                        group_sca[grp] = { 
                            'min': np.array(num_scalar),
                            'avg': np.array(num_scalar),
                            'max': np.array(num_scalar)
                        }
                        group_non[grp] = { 
                            'min': np.array(num_none),
                            'avg': np.array(num_none),
                            'max': np.array(num_none)
                        }
                        group_pra[grp] = {
                            'min': np.array(per_action),
                            'avg': np.array(per_action),
                            'max': np.array(per_action)
                        }
                        group_prs[grp] = { 
                            'min': np.array(per_scalar),
                            'avg': np.array(per_scalar),
                            'max': np.array(per_scalar)
                        }
                        group_prn[grp] = { 
                            'min': np.array(per_none),
                            'avg': np.array(per_none),
                            'max': np.array(per_none)
                        }
                    else:
                        ## Update running averages
                        n = group_num[grp] + 1
                        group_num[grp] = n
                        ## grp_avg contains the previous average
                        for k in ('min', 'max', 'end', 'ini', 'len'):
                            for idx, v in enumerate(grp_avg[k]):
                                group_avg[grp][k][idx] = (v*(n-1) + violations_dict[k][idx])/n
                            for idx, v in enumerate(group_gsr[grp][k]):
                                group_gsr[grp][k][idx] = (v*(n-1) + goal_success_dict[k][idx])/n
                        ## Update group effort statistics: Update average effort per trial
                        ## Trial effort includes only Action and Scalar feedback counts
                        for idx, v in enumerate(group_eff[grp]['avg']):
                            group_eff[grp]['avg'][idx] = (v*(n-1) + trial_efforts[idx])/n
                        for idx, v in enumerate(group_eff[grp]['min']):
                            group_eff[grp]['min'][idx] = min(v, trial_efforts[idx])
                        for idx, v in enumerate(group_eff[grp]['max']):
                            group_eff[grp]['max'][idx] = max(v, trial_efforts[idx])

                        ## Update Effort distribution: number of action feedback per trial
                        for idx, v in enumerate(group_act[grp]['avg']):
                            group_act[grp]['avg'][idx] = (v*(n-1) + effort_dist[cg.env.ACTION_FEEDBACK][idx])/n
                        for idx, v in enumerate(group_act[grp]['min']):
                            group_act[grp]['min'][idx] = min(v, effort_dist[cg.env.ACTION_FEEDBACK][idx])
                        for idx, v in enumerate(group_act[grp]['max']):
                            group_act[grp]['max'][idx] = max(v, effort_dist[cg.env.ACTION_FEEDBACK][idx])

                        ## Update Effort distribution: number of action feedback per trial
                        for idx, v in enumerate(group_pra[grp]['avg']):
                            group_pra[grp]['avg'][idx] = (v*(n-1) + per_action[idx])/n
                        for idx, v in enumerate(group_pra[grp]['min']):
                            group_pra[grp]['min'][idx] = min(v, per_action[idx])
                        for idx, v in enumerate(group_pra[grp]['max']):
                            group_pra[grp]['max'][idx] = max(v, per_action[idx])

                        ## Update Effort Distribution: number of scalar feedback per trial
                        for idx, v in enumerate(group_sca[grp]['avg']):
                            group_sca[grp]['avg'][idx] = (v*(n-1) + effort_dist[cg.env.SCALAR_FEEDBACK][idx])/n
                        for idx, v in enumerate(group_sca[grp]['min']):
                            group_sca[grp]['min'][idx] = min(v, effort_dist[cg.env.SCALAR_FEEDBACK][idx])
                        for idx, v in enumerate(group_sca[grp]['max']):
                            group_sca[grp]['max'][idx] = max(v, effort_dist[cg.env.SCALAR_FEEDBACK][idx])

                        ## Update Effort Distribution: number of scalar feedback per trial
                        for idx, v in enumerate(group_prs[grp]['avg']):
                            group_prs[grp]['avg'][idx] = (v*(n-1) + per_scalar[idx])/n
                        for idx, v in enumerate(group_prs[grp]['min']):
                            group_prs[grp]['min'][idx] = min(v, per_scalar[idx])
                        for idx, v in enumerate(group_prs[grp]['max']):
                            group_prs[grp]['max'][idx] = max(v, per_scalar[idx])

                        ## Update Effort Distribution: number of no feedback per trial
                        for idx, v in enumerate(group_non[grp]['avg']):
                            group_non[grp]['avg'][idx] = (v*(n-1) + effort_dist[cg.env.NO_FEEDBACK][idx])/n
                        for idx, v in enumerate(group_non[grp]['min']):
                            group_non[grp]['min'][idx] = min(v, effort_dist[cg.env.NO_FEEDBACK][idx])
                        for idx, v in enumerate(group_non[grp]['max']):
                            group_non[grp]['max'][idx] = max(v, effort_dist[cg.env.NO_FEEDBACK][idx])

                        ## Update Effort Distribution: number of no feedback per trial
                        for idx, v in enumerate(group_prn[grp]['avg']):
                            group_prn[grp]['avg'][idx] = (v*(n-1) + per_none[idx])/n
                        for idx, v in enumerate(group_prn[grp]['min']):
                            group_prn[grp]['min'][idx] = min(v, per_none[idx])
                        for idx, v in enumerate(group_prn[grp]['max']):
                            group_prn[grp]['max'][idx] = max(v, per_none[idx])
        ## Adjust the baseline (remove the first episode)
        if 'baseline' in group_avg:
            for k in ('min', 'max', 'end', 'ini', 'len'):
                group_avg['baseline'][k] = group_avg['baseline'][k][1:]
                group_gsr['baseline'][k] = group_gsr['baseline'][k][1:]
            for k in ('min', 'avg', 'max'):
                group_eff['baseline'][k] = group_eff['baseline'][k][1:]
                group_act['baseline'][k] = group_act['baseline'][k][1:]
                group_sca['baseline'][k] = group_sca['baseline'][k][1:]
                group_non['baseline'][k] = group_sca['baseline'][k][1:]
                group_pra['baseline'][k] = group_pra['baseline'][k][1:]
                group_prs['baseline'][k] = group_prs['baseline'][k][1:]
                group_prn['baseline'][k] = group_prs['baseline'][k][1:]
        plot_group_violations(
            group_avg, group_gsr, group_eff, 
            group_act, group_sca, group_non,
            group_pra, group_prs, group_prn,
            show=show_plots
        )

        exit()
    else:
        print(f"Mode must be train or test")
        exit()

    plt.rcParams.update({'font.size': 20})
    all_violations, detailed_violations = compute_violations(all_training_data, cg)
    plot_violations(all_violations, detailed_violations, args.dataset, show=show_plots)
    if demonstration_losses is not None:
        demo_losses = compute_demonstration_losses(all_training_data, cg, demonstration_losses)
        plot_losses(demo_losses, args.dataset, show=show_plots)

    distances = compute_distances(all_training_data, cg)
    for k, v in distances.items():
        plot_distances(v, k, args.dataset, show=show_plots)
