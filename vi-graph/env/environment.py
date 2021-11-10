from functools import reduce
import copy
from .world import Worlds
import torch
import numpy as np
from computation_graph import ComputationGraph
from grid_world_search import GridWorldProblem, GridWorldSearch

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
        self.action_feedback_map = {
            "w": 0,
            "d": 1,
            "s": 2,
            "a": 3,
            "stay": 4
        }

        self.SCALAR_FEEDBACK = 100
        self.ACTION_FEEDBACK = 200              ## Feedback is a single action, (NOT a 1-step s,a sequence)
        self.TRAJECTORY_FEEDBACK = 210          ## Feedback is a single trajectory (an N-step s,a sequence)
        self.TRAJECTORY_SET_FEEDBACK = 220      ## Feedback is a set of trajectories (a set of N-step s,a sequences)
        self.NO_FEEDBACK = 300

        self.need_simulated_evaluative_feedback = True
        self.sim_cg = None
        self.sim_advantage = None
        self.transition_matrix = None
        self.matrix_map = None
        self.rng = np.random.default_rng()

    def flattened_sas_transitions(self):
        """
        Returns 1D version of (s,a,s') map
        """
        nrows, ncols = self.world.shape
        state_idx = (lambda r, c: r*ncols + c)
        in_bounds = (lambda r, c: (0 <= r and r < nrows) and (0 <= c and c < ncols))
        transitions = dict()
        for r in range(nrows):
            for c in range(ncols):
                for a in range(len(self.actions)):
                    action = self.actions[a]
                    s = state_idx(r,c)
                    rp = r + action[0]
                    cp = c + action[1]
                    if in_bounds(rp, cp):
                        sp = state_idx(rp,cp)
                        if s not in transitions:
                            transitions[s] = {a: sp}
                        else:
                            transitions[s][a] = sp
        return transitions

    def all_sas_transitions(self, transitions):
        print(transitions)
        """
        Given a transitions dict, returns the possible tuples (s,a,s')
        """
        return tuple( tuple((s,a,sp))
            for s, asp in transitions.items()
            for a, sp in asp.items()
        )

    def get_world(self):
        return copy.deepcopy(self.world)

    def inform_human(self, trajectory):
        """
        The environment informs the human what action the agent is planning to take

        Takes in an action-state sequence, and displays the trajectory that the agent plans to take
        in the environment
        """
        print(f"The agent plans to take the following {len(trajectory)}-step trajectory:")
        nrows, ncols = self.world.shape
        grid = self.world + 6
        for s, a in trajectory:
            grid[s] = a

        for r in range(nrows):
            line = ""
            for c in range(ncols):
                line += '^>v<x?01234'[grid[r, c]]
            print(line)

    def request_feedback_from_human(self):
        """
        The agent will request a specific feedback type from the human
        """
        pass

    def acquire_feedback(self, trajectory, r, c, source_is_human=True, feedback_policy_type="action", agent_cg=None):
        """
        Acquires feedback from a source

        action_idx = index into the env.actions array such that env.actions[action_idx] is the index
            the agent plans on making
        r, c = row and colum position of the agent
        source_is_human = boolean indicating whether a human will give the feedback, or if an automated
            teacher will provide the feedback
        feedback_policy_type = if an automated teaching/feedback strategy is used, what is that strategy?
        agent_cg = for some teaching / feedback strategies, it's assumed the teacher has formed some estimate
            or model of the agent's policy, based on observing the agent's behavior over time. We pass in the
            the agent's computational graph in order to automated teaching strategies to simplify this modeling
            process.  If a teaching strategy should not have access to the agent's policy, then pass in None
        """
        def is_number(f):
            """
            This function is here to support raw numerical feedback (from teaching/feedback automation).
            """
            try:
                ffff = float(f)
            except ValueError:
                return False
            except TypeError:
                return False
            return True

        valid_feedback = {"-2","-1","0","1","2","w","a","s","d","stay", "x"}
        feedback_str = None
        valid = True

        ## TODO: Need to make the following changes: all action -> trajectory (in terms of feedback)

        while feedback_str not in valid_feedback and not is_number(feedback_str):
            if source_is_human:
                if not valid: print("Invalid feedback specified")
                print("Feedback Options: Action Options:  w(UP), d(RIGHT), s(DOWN), a(LEFT), stay(STAY)")
                print("Feedback Options: Scalar Options:  -2,  -1,  0,  1,  2")
                feedback_str = input("Human Feedback: ")
            else:
                return self.feedback_source(trajectory, r, c, feedback_policy_type, agent_cg)
            valid = False
        return feedback_str

    def classify_feedback(self, feedback):
        """
        Classifies the feedback into a feedback category
        """
        print(f"Recieved: {feedback}")
        if feedback in "-2 -1 0":
            return self.SCALAR_FEEDBACK
        else:
            try:
                f = float(feedback)
                return self.SCALAR_FEEDBACK
            except ValueError:
                ## Couldn't convert feedback to a float. Assuming ACTION_FEEDBACK
                return self.ACTION_FEEDBACK

    def feedback_to_scalar(self, feedback):
        """
        Converts feedback to scalar
        """
        return float(feedback)

    def feedback_to_demonstration(self, feedback, r, c):
        """
        Converts feedback to trajact and trajcoord
        """
        return [self.action_feedback_map[feedback]], [(r,c)]

    def __optimal_trajectory_from_state(self, state, max_length=1, feedback_map=None):
        assert feedback_map is not None
        trajectory_length = max_length
        feedback_trajectory = []
        r_s, c_s = state
        for _ in range(trajectory_length):
            s = (r_s, r_c)
            a_str = feedback_map[r_s][c_s]
            act = self.action_feedback_map[a_str]
            feedback_trajectory.append((s, act))
            r_s, r_c = self.step(r_s, r_c, act)
        return feedback_trajectory

    def feedback_source(self, trajectory, r, c, feedback_policy_type, agent_cg):
        """
        A simulator for providing feedback.
        This function is called from acquire_feedback()

        trajectory - sequences of (s_0, a)
        r, c - gridworld coordinates row, col
        feedback_policy_type - string, this comes in as a program argument
        agent_cg = for some teaching / feedback strategies, it's assumed the teacher has formed some estimate
            or model of the agent's policy, based on observing the agent's behavior over time. We pass in the
            the agent's computational graph in order to automated teaching strategies to simplify this modeling
            process.  If a teaching strategy should not have access to the agent's policy, then pass in None

        This will now return (feedback, feedback_type)

        NOTE: If a full trajectory is provided, then all feedback should be with respect to the trajectory.
            - If action feedback will be given, then provide an optimal trajectory of up to equal length of the
                input trajectory.
            - If evaluative feedback will be given, then...

        """
        nrows, ncols = self.world.shape
        ## TODO: Implement a feedback simulator to simulate giving feedback
        ## NOTE: For now, we've implemented optimal feedback for each grid location
        ## Optimal actions map for only world 1
        feedback_map = (
            ("s","s","a","d","s","s","s","s","d","stay"),
            ("s","s","s","d","d","d","d","d","d","w"),
            ("d","d","d","d","d","d","d","d","d","w"),
            ("w","w","w","w","w","w","w","d","d","w"),
            ("w","w","a","a","a","w","w","d","d","w")
        )
        if feedback_policy_type == "action":
            return self.action_feedback_map[feedback_map[r][c]], self.ACTION_FEEDBACK

        if feedback_policy_type == "1_step_corrective":
            """
            Ideally generalize this to optimal actions for arbitrary worlds other than world 1
            Currently, this is relegated to single time-step trajectories

            For trajectory-feedback, there are two options:
            If the agent provides an N-step trajectory, there are N states in that trajectory (some of the N-states may be very undesireable)
              (a) Provide optimal action at each of the N-states in the trajectory; this is "corrective"
              (b) Provide optimal trajectory starting from the initial state of the trajectory; this is a "demonstration"
              (c) Provide optimal trajectory starting from each of the N-states in the trajectory; this provides "corrective demonstrations"
              (d) If human had to choose a single most informative trajectory from the set in (c), which one would be chosen?
            (The first option seems to be the 1-step version of the third option).

            For "action" we inherit the "corrective" feedback style (a), so this is a "series of optimal 1-step trajectories" from each state
            which is just a special case of (c)
            """
            trajectory_length = len(trajectory)
            feedback_trajectory = []
            for s, a in trajectory:
                r_s, c_s = s
                a_str = feedback_map[r_s][c_s]
                act = self.action_feedback_map[a_str]
                feedback_trajectory.append((s, act))
            return feedback_trajectory, self.TRAJECTORY_FEEDBACK

        if feedback_policy_type == "init_demonstration":
            """
            This implements strategy (b) Provide optimal trajectory starting from the initial state of the trajectory
            """
            feedback_trajectory = self.__optimal_trajectory_from_state((r,c), max_length=len(trajectory), feedback_map=feedback_map)
            return feedback_trajectory, self.TRAJECTORY_FEEDBACK

        if feedback_policy_type == "set_demonstration":
            """
            This implements strategy (c): teacher provides back a set of N optimal demonstrations
            """
            trajectory_length = len(trajectory)
            feedback_trajectory = []
            adj = 0
            for s, a in trajectory:
                f_trajectory = self.__optimal_trajectory_from_state(s, max_length=trajectory_length - adj, feedback_map=feedback_map)
                feedback_trajectory.append((s, f_trajectory))
                adj = adj + 1
            return feedback_trajectory, self.TRAJECTORY_SET_FEEDBACK
            
        if feedback_policy_type == "selective_demonstration":
            """
            This implements strategy (d): from the set computed in (c), teacher selects a demonstration to return, according to criteria.
            Ideally the teacher should select the trajectory that provides the most information to the agent?

            Much of the literature uses a softmax of trajectory returns to obtain a probabilistic value to argmax over. Why?
            """
            raise NotImplementedError("selective_demonstration has not been implemented")
            pass

        if feedback_policy_type == "r1_evaluative":
            """
            A very rough automated approximation to researcher 1's evaluative feedback strategy.
            This is based on ranking of path-cost, and then making adjustments for violations and other
            undesirable behaviors.

            This first solves a for optimal-cost path via uniform-cost-search for every position in the grid
            world to the goal
            """
            ## i.e. researcher 1's evaluative feedback teaching strategy:
            ## Whenever a violation is made, the feedback must be negative. Feedback is a scalar that is ranked
            ## and given such that the action brings the agent closer to the goal (i.e. along a shortest path)
            ## is considered the best.
            if self.need_simulated_evaluative_feedback:
                goal_state = tuple(np.argwhere(self.world == 4)[0])
                prob = GridWorldProblem(self, goal_state, goal_state)
                search_problem = GridWorldSearch(prob)
                costs, paths = search_problem.cost_to_goal()
                self.need_simulated_evaluative_feedback = False
                ## NOTE: These are the base costs from each point in the grid to the goal IF taking the shortest
                ## path. When decision making, need to add the cost of transitioning from the current state
                ## TO one of these cells to determine the actual cost of taking an action
                ## Example for world 1:
                ##   [[ 13.  12.  13.   8.   7.   6.   5.   4.   1.   0.]
                ##    [ 12.  11.  10.   7.   6.   5.   4.   3.   2.   1.]
                ##    [ 11.  10.   9.   8.   7.   6.   5.   4.   3.   2.]
                ##    [ 12.  11.  10.   9.   8.   7.   6.   5.   4.   3.]
                ##    [ 13.  12.  13.  14.  15. 107. 106.   6.   5.   4.]]
                ##
                self._costs = costs
                self._paths = paths
                prob.set_forward(forward=True)
                self._problem = prob

            ## NOTE: Feedback is "advantage-based" in this scenario in that the (a,s') pairs are ranked according to
            ## the lowest cost of s'.
            ## The possible actions are given by self.env.actions
            ## NOTE: Decision costs are computed in the following way:
            ##   The agent is currently at (r,c) and will take an action a that takes it to (r',c')
            ##   Ranking of the action is determined by self._costs[(r',c')] + transition_cost((r,c), a, (r',c'))
            decision_costs = np.zeros(len(self.actions))
            for idx in range(len(self.actions)):
                a = self.actions[idx]
                next_state = self._problem.transition((r,c), a)
                edge_cost = self._problem.transition_cost((r,c), a, next_state)
                decision_costs[idx] = self._costs[next_state] + edge_cost
            ## NOTE: argsort does highest value = largest index; lowest value = lowest index
            ## In the lowest cost case, we want the highest index to have highest advantage
            ## For example: 
            ## If decision costs is : [101.   1.   3. 104. 101.]
            ## then the argsort is [1, 2, 0, 4, 3]  ** See the np.argsort() documentation
            ## To get the rank, do .argsort() again
            mapped_advantage = (decision_costs.argsort().argsort() - 2)*(-1)
            ## Addtional adjustments on "0" advantage cases:
            ## "Staying Put" has ranking of 0, then it should get -2 advantage
            ## "Violating action" has ranking of 0, give it -2
            zero_idx = np.where(mapped_advantage == 0)[0][0]
            ## Give staying put a negative value
            if mapped_advantage[4] == 0:
                mapped_advantage[4] -= 2
                if mapped_advantage[4] < -2:
                    mapped_advantage[4] = -2
            ## Transitions to a violating state should be negative
            if decision_costs[zero_idx] > 100:
                mapped_advantage[zero_idx] = -2
            ## Transitions that yield a larger min cost than current should be negative
            if decision_costs[zero_idx] < 100:
                ## If the action brings it to a state where the min cost is greater than the current state,
                ## then make sure that gets a negative value
                current_cost = self._costs[(r,c)]
                other_cost = self._costs[self._problem.transition((r,c), self.actions[zero_idx])]
                if other_cost > current_cost:
                    mapped_advantage[zero_idx] = -2
            print(decision_costs)
            print(mapped_advantage)
            ## Fix for giving 0 advantage to "staying put"
            #if action == 4 and mapped_advantage[action] == 0:
            #    return str(-2)
            return str(mapped_advantage[action]), self.SCALAR_FEEDBACK

        if feedback_policy_type == "ranked_evaluative":
            """
            "Ranks" the advantage values computed from a ground truth reward function
            i.e. runs "raw_evaluative" to get the advantage values, and then maps them onto
            a {-2, -1, 0, 1, 2} range
            This version is independent of the agent's policy
            """
            self.__compute_ground_truth()
            ## Consider just advantage mapped onto -2, -1, +0, +1, +2 via ranking.
            ## NOTE: Maps to -2, -1, +0, +1, +2 evaluative ranking that we have been using for experiments.
            ## If we don't want to do this, then just return the raw advantage
            ## To return raw advantage, set map_to_ranked_advantage_scale to False
            mapped_advantage = self.sim_advantage.argsort().argsort() - 2
            #print(mapped_advantage)
            #print(f"r: {r}, c: {c}, nrows: {nrows}, action: {action}")
            return str(mapped_advantage[r*nrows + c, action]), self.SCALAR_FEEDBACK
       
        if feedback_policy_type == "policy_evaluation":
            """
            The teacher evaluates an estimate of the agent's policy in order to provide feedback.
            For now, we provide the agent's exact policy, and evaluate it against a ground-truth
            reward function.  Returns raw advantage values.

            This evaluates the agent's policy against a ground truth reward function (hence making
            the feedback under this strategy policy dependent, inline with COACH).
            """
            agent_pi, agent_Q, agent_V = agent_cg.forward(as_numpy=True)
            self.__compute_ground_truth()
            A, Q, V = self.policy_evaluation(self.sim_cg.current_reward_estimate(), agent_pi)
            raw_advantage = A[r*nrows+c][action]
            print(f"Agent Policy: {agent_pi[r*nrows+c]}")
            print(f"A: {A[r*nrows+c]}")
            print(f"Q: {Q[r*nrows+c]}")
            print(f"V: {V[r*nrows+c]}")
            return str(raw_advantage), self.SCALAR_FEEDBACK

        if feedback_policy_type == "ranked_policy_evaluation":
            """
            The teacher evaluates an estimate of the agent's policy in order to provide feedback.
            For now, we provide the agent's exact policy, and evaluate it against a ground-truth
            reward function.  Returns raw advantage values.

            This evaluates the agent's policy against a ground truth reward function (hence making
            the feedback under this strategy policy dependent, inline with COACH).
            """
            if action == 4:
                return str(-2) ## Discourage "staying"
            agent_pi, agent_Q, agent_V = agent_cg.forward(as_numpy=True)
            self.__compute_ground_truth()
            A, Q, V = self.policy_evaluation(self.sim_cg.current_reward_estimate(), agent_pi)
            raw_advantage = A[r*nrows+c][action]
            print(f"Agent Policy: {agent_pi[r*nrows+c]}")
            print(f"A: {A[r*nrows+c]}")
            print(f"Q: {Q[r*nrows+c]}")
            print(f"V: {V[r*nrows+c]}")
            ranked_advantage = A.argsort().argsort() - 2
            ## Set 0 to -1
            ranked_advantage[ranked_advantage == 0] = -1
            return str(ranked_advantage[r*nrows+c, action]), self.SCALAR_FEEDBACK
 
        if feedback_policy_type == "raw_evaluative":
            """
            Uses the raw advantage values computed from a ground truth reward function
            This version is independent of the agent's policy
            """
            self.__compute_ground_truth()
            ## TODO: Support raw advantage values (for arbitrary scaling)
            ## Alternatively, we should also be able to handle direct advantage feedback
            ## Scalar only
            ## Return advantage value
            ## Assumes a reward function that yields correct optimal behavior
            ## However, what's interesting that is that depending on the scale,
            ## the advantage values will be different! So maybe normalize the values.
            raw_advantage = self.sim_advantage[r*nrows+c][action]
            print(f"Advantage: {self.sim_advantage[r*nrows+c]}")
            print(f"Policy: {self.policy[r*nrows+c]}")
            print(f"QVals: {self.Qvals[r*nrows+c]}")
            print(f"Vvals: {self.Vvals[r*nrows+c]}")
            #print(self.sim_advantage)
            ## NOTE: This is not yet supported
            #raise ValueError("Raw advantage support not yet implemented.")
            return str(raw_advantage), self.SCALAR_FEEDBACK

        if feedback_policy_type == "mixed":
            """
            This is supposed to represent a placeholder for some mix of action and evaluative feedback
            """
            ### Create a static mixture of the action and evaluative feedback
            ### Create a dynamic mixture of action and evaluative feedback (distribution changes over time)
            ### Create a mixture dependent on the history of the state, action, feedback encountered
            if self.rng.uniform() < 0.5:
                return self.feedback_source(action, r, c, "action")
            else:
                return self.feedback_source(action, r, c, "evaluative")

    def policy_evaluation(self, reward_func, policy):
        """
        Evaluates a policy under a ground truth reward function.
        reward_func = a feature-based reward function to evaluate the policy against
            numpy_array or pytorch.tensor
        policy = a policy to evaluate against the reward (i.e. find V and Q-values under this policy)
            numpy_array or pytorch.tensor
        """
        ## Solve system of equations to determine V^pi, or just iterate Bellman equation
        ## policy = |S| x |A|
        stochastic_pi = torch.tensor(policy, dtype=self.dtype, requires_grad=False)
        ## Transition matrix is |A|x|S|x|S'|
        if self.transition_matrix is None:
            self.transition_matrix = self.get_mattrans()
        if self.matrix_map is None:
            self.matrix_map = self.get_matmap()

        ## reward func is |F|-length vector. we want to compute R(s,a), which is the expected reward you get
        ## for taking action (a) in state (s) (and leading you into state s'). We can construct this from the
        ## transition matrix: R(s,a) = T(a,s,s')R(s'), then sum reduce across the S' dimension (i.e. take inner product)
        ## This will result in a |A|x|S| reward matrix
        r_s = self.__compute_reward_s(reward_func)
        r_sa = self.__compute_reward_sa(r_s)
        ## one step expected return: |S| element vector:
        ## one_step = \sum_a \pi(a|s)r(s,a)
        one_step = (stochastic_pi*(r_sa.T)).sum(-1)
        
        ## Initialize the Value function to this
        V = r_s.detach().clone()

        g = 0.99 ## gamma
        for _ in range(100):
            ## Bellman update
            ## V^\pi(s) = \sum_a \pi(a|s)r(s,a) + g\sum_a \pi(a|s) \sum_{s'} T(s'|s,a)V^\pi(s')
            V = one_step + g*((stochastic_pi*(torch.matmul(self.transition_matrix, V).T)).sum(-1))

        ## We can compute Q^\pi from V^\pi, T
        ## Q^\pi(s,a) = r(s,a) + g\sum_{s'} T(s'|s,a)V^\pi(s')
        ## Do transpose to get a |S|x|A| matrix
        Q = (r_sa + g*torch.matmul(self.transition_matrix, V)).T
        A = self.__compute_advantage(Q, V.detach().clone())
        return A.numpy(), Q.numpy(), V.numpy()

    def __compute_reward_sa(self, rs):
        """
        Computes a R(s,a) reward function using R(s') and T(a,s,s')
        """
        if self.transition_matrix is None:
            self.transition_matrix = self.get_mattrans()
        ## If  |A|x|S|x|S'| x |S'|, then do torch.matmul(T, R)
        ## If  |A|x|S|x|S'| x |S'|x1, then do torch.matmul(T, R.squeeze()) or torch.matmul(T, R).sum(-1) or torch,matmul(T,R).squeeze()
        r_sa = torch.matmul(self.transition_matrix, rs)  ## should be |A| x |S|
        return r_sa

    def __compute_reward_s(self, reward_func):
        """
        Computes a R(s) reward function using the feature-based reward function and a matrix map
        """
        if self.matrix_map is None:
            self.matrix_map = self.get_matmap()
        rk = torch.tensor(reward_func, dtype=self.dtype, requires_grad=False)
        new_rk = rk.unsqueeze(0)
        new_rk = new_rk.unsqueeze(0)
        nrows, ncols, ncategories = self.matrix_map.shape
        new_rk = new_rk.expand(nrows, ncols, ncategories)
        ## Dot product to obtain the reward function applied to the matrix map
        rfk = torch.mul(self.matrix_map, new_rk)
        rfk = rfk.sum(axis=-1)
        rffk = rfk.view(nrows*ncols) ## 1D view of the 2D grid
        #initialize the value function to be the reward function, required_grad should be false
        return rffk

    def __compute_advantage(self, Q, V):
        """
        Computes A = Q-V

        Expects Q = |S|x|A| (2-D), V = |S| (1-D)
        """
        V = V.unsqueeze(0).T.expand(Q.shape)
        ## Advantage
        A = Q - V ## 2D nrows*ncols x nacts
        return A

    def __compute_ground_truth(self):
        """
        This computes an optimal policy from a feature-based ground truth reward function.
        """
        if self.sim_advantage is None or self.need_simulated_evaluative_feedback:
            ## We haven't computed the simulated advantage values yet,
            ## so compute those values using a prespecified reward function
            self.sim_cg = ComputationGraph(self)
            ## Specify a reward function to use
            ## default_0
            ## DEFAULT_REWARD = [0, -1, -1, -1,  1]
            ## default_1
            ## DEFAULT_REWARD = [1.3605843, -1.1135638, -0.62023115, -1.9482319, 2.837054]
            ## default_2
            ## DEFAULT_REWARD = [ 0.6487329, -2.6717327, -1.280116, 0.519958, 2.7245345]
            ## default_3 (from supervised action-only version, probably most accurate?)
            DEFAULT_REWARD = [ 0.5631, -0.4677, -0.4628, -0.7351,  1.1025]
            reward = np.array(DEFAULT_REWARD) - DEFAULT_REWARD[0] - 0.01
            reward = reward / np.linalg.norm(reward)
            self.sim_cg.set_reward_estimate(reward)
            pi, Q, V = self.sim_cg.forward()
            ## Q is (nrows*ncols) x (nacts) (2D)
            ## V is (nrows*ncols) (1D)
            V = V.unsqueeze(0).T.expand(Q.shape)
            ## Advantage
            A = (Q - V).cpu().detach().numpy() ## 2D nrows*ncols x nacts
            self.need_simulated_evaluative_feedback = False
            self.sim_advantage = A
            self.policy = pi
            self.Qvals = Q
            self.Vvals = V
            #self.sim_cg = None

    def random_start_state(self):
        """
        Select a random start state (chooses a random (r,c) that is a 0 colored square)
        This is now a generator.
        """
        indices = np.flatnonzero(self.world == 0)
        total = len(indices)
        _, ncols = self.world.shape
        n = 0
        while True:
            if n == 0:
                np.random.shuffle(indices)
            idx = indices[n]
            r = idx // ncols
            c = idx %  ncols
            yield r, c
            n = (n + 1) % total

    def get_termination_states(self):
        """
        Return the location of the termination state
        """
        pass

    def get_goal_state(self):
        """
        Returns the color that is considered a goal
        """
        return 4

    def get_start_state(self):
        return copy.copy(self.state_start)

    def step(self, r, c, action):
        """
        Transitions r,c based on action
        """
        rr, cc = self.actions[action]
        return (r + rr, c + cc)

    def get_single_timestep_action(self):
        """
        Retrieves a single discrete timestep action from a human
        """
        k = input("Action: ")
        if k not in self.action_feedback_map:
            return -1
        else:
            return self.action_feedback_map[k]

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
        def clip(v,min_v,max_v):
            if v < min_v: v = min_v
            if v > max_v-1: v = max_v-1
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
