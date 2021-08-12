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
        self.action_feedback_map = {
            "w": 0,
            "d": 1,
            "s": 2,
            "a": 3,
            "stay": 4
        }

        self.SCALAR_FEEDBACK = 100
        self.ACTION_FEEDBACK = 200
        self.NO_FEEDBACK = 300

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

    def inform_human(self, action_idx):
        """
        The environment informs the human what action the agent is planning to take
        """
        print(f"The agent plans to do action: {self.act_name[action_idx]}")

    def request_feedback_from_human(self):
        """
        The agent will request a specific feedback type from the human
        """
        pass

    def acquire_feedback(self, action_idx, r, c, source_is_human=True):
        """
        Acquires feedback from a source
        """
        valid_feedback = {"-2","-1","0","1","2","w","a","s","d","stay", "x"}
        feedback_str = None
        valid = True
        while feedback_str not in valid_feedback:
            if source_is_human:
                if not valid: print("Invalid feedback specified")
                print("Feedback Options: Action Options:  w(UP), d(RIGHT), s(DOWN), a(LEFT), stay(STAY)")
                print("Feedback Options: Scalar Options:  -2,  -1,  0,  1,  2")
                feedback_str = input("Human Feedback: ")
            else:
                feedback_str = self.feedback_source(action_idx, r, c)
            valid = False
        return feedback_str

    def classify_feedback(self, feedback):
        """
        Classifies the feedback into a feedback category
        """
        if feedback in "-2 -1 0":
            return self.SCALAR_FEEDBACK
        else:
            return self.ACTION_FEEDBACK

    def feedback_to_scalar(self, feedback):
        """
        Converts feedback to scalar
        """
        return int(feedback)

    def feedback_to_demonstration(self, feedback, r, c):
        """
        Converts feedback to trajact and trajcoord
        """
        return [self.action_feedback_map[feedback]], [(r,c)]

    def feedback_source(self, action, r, c):
        """
        A simulator for providing feedback
        """
        ## TODO: Implement a feedback simulator to simulate giving feedback
        ## NOTE: For now, we've implemented optimal feedback for each grid location
        feedback_map = (
            ("s","s","a","d","s","s","s","s","d","stay"),
            ("s","s","s","d","d","d","d","d","d","w"),
            ("d","d","d","d","d","d","d","d","d","w"),
            ("w","w","w","w","w","w","w","d","d","w"),
            ("w","w","a","a","a","w","w","d","d","w")
        )
        return feedback_map[r][c]

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
