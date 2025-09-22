from pomdp_py.framework.basics cimport *
from pomdp_py.framework.planner cimport Planner
from pomdp_py.representations.distribution.particles cimport Particles
from pomdp_py.utils import typ

import time
import numpy as np
import copy
from tqdm import tqdm
import scipy

cdef class TreeNode:
    def __init__(self):
        self.children = {}

    def __getitem__(self, key):
        return self.children.get(key, None)

    def __setitem__(self, key, value):
        self.children[key] = value

    def __contains__(self, key):
        return key in self.children

cdef class QNode(TreeNode):
    """A history-action node in the search tree."""

    def __init__(self, num_visits=0, R=0.0, D=0.0, pref=0.0):
        super().__init__()
        self.num_visits = num_visits
        self.R = R
        self.D = D
        self.pref = pref
        self.children = {}

    def __str__(self):
        return typ.red(f'QNode<N:{self.num_visits}, R:{self.R}, D:{self.D}, Pref:{self.pref} | {self.children.keys()}>')

    def __repr__(self):
        return self.__str__()

cdef class VNode(TreeNode):
    """A history node in the search tree."""
    def __init__(self, num_visits=0, V=0.0):
        super().__init__()
        self.num_visits = num_visits
        self.V = V
        self.children = {}

    def __str__(self):
        return f'VNode<N:{self.num_visits}, V:{self.V} | num_children: {len(self.children)}>'

    def __repr__(self):
        return self.__str__()

    def children_data(self):
        string = ""
        for action in self.children:
            string += f' |_{action}, N:{self[action].num_visits}, R:{round(self[action].R, 1)}, Pref:{round(self[action].pref, 1)}\n'
        return string

    cpdef least_visited(VNode self):
        "Returns the action of the child with the lowest number of visits."
        cdef Action action, best_action
        cdef float min_num_visits = float("inf")
        for action in self.children:
            if self[action].num_visits < min_num_visits:
                best_action = action
                min_num_visits = self[action].num_visits
        if best_action is None:
            print("Warning: The highest preference is still -inf. Returned action is not meaningful.")
        return best_action

    cpdef argmax(VNode self):
        """argmax(VNode self)
        Returns the action of the child with highest preference"""
        cdef Action action, best_action
        cdef float best_pref = float("-inf")
        best_action = None
        for action in self.children:
            if self[action].pref >= best_pref:
                best_action = action
                best_pref = self[action].pref
        if best_action is None:
            for action in self.children:
                if self[action].pref >= best_pref:
                    best_action = action
                    best_pref = self[action].pref
        if best_action is None:
            print("Warning: The highest preference is still -inf. Returned action is not meaningful.")
        return best_action

    cpdef mellowmax(VNode self, float temperature):
        """The logsumexp of the child preferences."""
        return scipy.special.logsumexp([self[a].pref for a in self.children], b=temperature) / temperature

    cpdef boltzmann(VNode self, float temperature):
        """The Boltzmann soft-max of the child preferences."""
        cdef list preferences, input
        preferences = [self[a].pref for a in self.children]
        input = [pref * temperature for pref in preferences]
        return np.dot(preferences, scipy.special.softmax(input))

    cpdef sample_softmax(VNode self, float temperature, str value):
        """Sample an action based on the softmax of the:
            1) preferences;
            2) number of visits."""
        if temperature == np.inf:
            return self.argmax()
        cdef list input
        if value == "pref":
            input = [self[a].pref * temperature for a in self.children]
        elif value == "num_visits":
            input = [-self[a].num_visits * temperature for a in self.children]
        else:
            raise ValueError("Unrecognized string for value.")
        return np.random.choice([a for a in self.children],
                                size=1, replace=True, p=scipy.special.softmax(input))[0]

    @property
    def max_pref(self):
        best_action = self.argmax()
        return self.children[best_action].pref

cdef class VNodeParticles(VNode):
    """A history node with a particle representation of the associated belief."""

    def __init__(self, num_visits=0, V=0.0, belief=Particles([])):
        self.num_visits = num_visits
        self.V = V
        self.belief = belief
        self.children = {}

    def __str__(self):
        return f'VNode<N:{self.num_visits}, V:{self.V}, Num Particles:{len(self.belief)} | num_children: {len(self.children)}>'

    def __repr__(self):
        return self.__str__()

cdef class RootVNode(VNode):
    def __init__(self, num_visits, history):
        VNode.__init__(self, num_visits)
        self.history = history

    @classmethod
    def from_vnode(cls, vnode, history):
        """from_vnode(cls, vnode, history)"""
        rootnode = RootVNode(vnode.num_visits, history)
        rootnode.children = vnode.children
        return rootnode

cdef class RootVNodeParticles(RootVNode):
    def __init__(self, num_visits, history, belief=Particles([])):
        RootVNode.__init__(self, num_visits, history)
        self.belief = belief

    @classmethod
    def from_vnode(cls, vnode, history):
        rootnode = RootVNodeParticles(vnode.num_visits, history, belief=vnode.belief)
        rootnode.children = vnode.children
        return rootnode

cdef class RGPOMCP(Planner):

    """
    Args:
        planning_time (float), amount of time given to each planning step (seconds). Default: -1.
            if negative, then planning terminates when number of simulations `num_sims` reached.
            If both `num_sims` and `planning_time` are negative, then the planner will test_utils for 1 second.
        num_sims (int): Number of simulations for each planning step. If negative,
            then will terminate when planning_time is reached.
            If both `num_sims` and `planning_time` are negative, then the planner will test_utils for 1 second.
        max_depth (int): Maximum single-step depth of the search tree. Default: 30.
            rollout_depth (int): The maximum depth of a rollout (Default: 50)
        discount_factor (float): The single-step discount factor. Default: 0.99.
        temperature (float): A positive real number indicating the temperature of the soft-max. Default: 5.
        pref_init (float): The default action preference. Default: 0.
        action_branching_factor (int): The maximum number of actions per node. Default: 10.
        exploitation_prob (float): Probability of exploiting versus exploring. Default: 0.5.
        exploitation_temp (float): Soft-max temperature for exploiting an action over preferences. Default: 0.05.
        exploration_const (float): UCB exploration const. Default: 1000.
        num_sir_particles (int): The number of particles to maintain as part of SIR belief update. Default: 1000.
        sir_temp (int): A positive real number indicating the temperature of the softmax used to exaggerate
            resampling weights. Default: 1.
        sir_zero_weight_exponent (float): A negative float x for which exp(x) re-assigns the weight for resampling a
            particle that has an effective resampling weight of zero. Default: -Inf.
        reward_ccale (float): A scale factor for the reward. Default: 1.0.
        reseed_global_objective (bool): Reseed global objective after every simulation. Default: True.
        show_progress (bool): True if print a progress bar for simulations.
        pbar_update_interval (int): The number of simulations to test_utils after each update of the progress bar,
            Only useful if show_progress is True; You can set this parameter even if your stopping criteria
            is time.
    """

    def __init__(self,
                 planning_time=-1., num_sims=-1,
                 max_depth=30, rollout_depth=50,
                 discount_factor=0.99,
                 temperature=1.,
                 pref_init=0.,
                 action_branching_factor=10,
                 exploitation_prob=0.5,
                 exploitation_temp=0.05,
                 exploration_const=1000,
                 num_sir_particles=1000,
                 sir_temp=1.,
                 sir_zero_weight_exponent=-np.inf,
                 reward_scale=1.,
                 reseed_global_objective=True,
                 show_progress=False, pbar_update_interval=5):
        self._planning_time = planning_time
        self._num_sims = num_sims
        if self._num_sims < 0 and self._planning_time < 0:
            self._planning_time = 1.
        self._max_depth = max_depth
        self._rollout_depth = rollout_depth
        self._discount_factor = discount_factor
        self._temperature = temperature
        self._pref_init = pref_init
        self._action_branching_factor = action_branching_factor
        self._exploitation_prob = exploitation_prob
        self._exploitation_temp = exploitation_temp
        self._exploration_const = exploration_const
        self._num_sir_particles = num_sir_particles
        self._sir_temp = sir_temp
        self._sir_zero_weight_exponent = sir_zero_weight_exponent
        self._reward_scale = reward_scale

        self._reseed_global_objective = reseed_global_objective
        self._global_objective = None

        self._show_progress = show_progress
        self._pbar_update_interval = pbar_update_interval

        # to simplify function calls; plan only for one agent at a time
        self._agent = None
        self._last_num_sims = -1
        self._last_planning_time = -1

    def __str__(self):
        return f'RGPOMCP Params[' \
               f'planning_time:{self._planning_time:.3f}, ' \
               f'num_sims:{self._num_sims}, ' \
               f'max_depth:{self._max_depth}, ' \
               f'rollout_depth:{self._rollout_depth}, ' \
               f'discount_factor:{self._discount_factor:.3f}, ' \
               f'temperature:{self._temperature:.3f}, ' \
               f'pref_init:{self._pref_init:.3f}, ' \
               f'action_branching_factor:{self._action_branching_factor:.3f}, ' \
               f'exploitation_prob:{self._exploitation_prob:.3f}, ' \
               f'exploitation_temp:{self._exploitation_temp:.3f}, ' \
               f'exploration_const:{self._exploration_const:.3f}, ' \
               f'num_sir_particles:{self._num_sir_particles}, ' \
               f'sir_temp:{self._sir_temp}, ' \
               f'sir_zero_weight_exponent:{self._sir_zero_weight_exponent}, ' \
               f'reward_scale:{self._reward_scale}, ' \
               f'reseed_global_objective:{self._reseed_global_objective}]'

    def __repr__(self):
        return self.__str__()

    @property
    def update_agent_belief(self):
        """True if planner's update function also updates agent's
        belief."""
        return True

    @property
    def last_num_sims(self):
        """Returns the number of simulations ran for the last `plan` call."""
        return self._last_num_sims

    @property
    def last_planning_time(self):
        """Returns the amount of time (seconds) ran for the last `plan` call."""
        return self._last_planning_time

    @property
    def max_depth(self):
        return self._max_depth

    @property
    def discount_factor(self):
        return self._discount_factor

    @property
    def global_objective(self):
        return self._global_objective

    cpdef public plan(self, Agent agent):
        cdef Action action
        cdef float time_taken
        cdef int sims_count

        self._agent = agent   # switch focus on planning for the given agent
        if not hasattr(self._agent, "tree"):
            self._agent.add_attr("tree", None)
        action, time_taken, sims_count = self._search()
        self._last_num_sims = sims_count
        self._last_planning_time = time_taken
        return action

    cpdef update(self, Agent agent, Action real_action, Observation real_observation,
                 state_transform_func=None):
        """
        Use sampling importance resampling to update the belief to reduce the likelihood 
        of particle deprivation. Particle reinvigoration is used to ensure a rich set of 
        particles. For technical details refer to the textbook by (Thrun, 2002).

        Assumes self._agent.observation_model.probability(o, next_s, a) is implemented.
        """
        cdef list particles
        cdef list existing_particles
        cdef int num_new_particles

        if not isinstance(agent.belief, Particles):
            raise TypeError("Agent's belief is not represented by particles.\n"\
                            "RGPOMCP not usable. Please convert it to particles.")

        if not hasattr(agent, "tree") or agent.tree is None:
            print("Warning: agent does not have tree. Have you planned yet?")
            return

        # Always perform resampling. This is because the continuous action space
        # means that the executed action may not be in the tree for very sparsely
        # separated beliefs.

        if agent.tree[real_action][real_observation] is None:
            existing_particles = []
            num_new_particles = self._num_sir_particles
        else:
            existing_particles = agent.tree[real_action][real_observation].belief.particles
            num_new_particles = max(self._num_sir_particles - len(existing_particles), 0)

        particles = agent.tree.belief.particles

        if isinstance(real_action, MacroAction):
            if not isinstance(real_observation, MacroObservation):
                raise ValueError('Action was a MacroAction. Expecting a MacroObservation too.')

            # sampling importance resampling
            for a, o in zip(real_action.action_sequence, real_observation.observation_sequence):
                particles = self._sir(agent, a, o, particles,
                                      self._sir_temp, num_new_particles)
        else:
            particles = self._sir(agent, real_action, real_observation, particles,
                                  self._sir_temp, num_new_particles)

        if agent.tree[real_action][real_observation] is None:
            print("Unanticipated history during update. Creating new root node.")
            agent.tree[real_action][real_observation] = VNodeParticles(belief=Particles(copy.deepcopy(particles)))
        else:
            print(f"Particles in belief node without resampling: {len(agent.tree[real_action][real_observation].belief)}")
            agent.tree[real_action][real_observation].belief = Particles(copy.deepcopy(particles + existing_particles))

        # Update the tree and use the resampled belief.
        agent.tree = RootVNodeParticles.from_vnode(agent.tree[real_action][real_observation],
                                                   agent.history)

        agent.set_belief(agent.tree.belief)

        print(f"Particles in belief node after resampling: {len(agent.tree.belief)}")

    cdef list _sir(self, Agent agent, Action real_action, Observation real_observation,
                   list particles, float temperature, int num_sir_particles):

        cdef float raw_weight
        cdef list weights
        cdef list updated_particles
        cdef State next_s
        weights = []
        updated_particles = []

        for s in particles:
            # sample next stage particles
            next_s = agent.transition_model.sample(s, real_action)
            raw_weight = agent.observation_model.probability(real_observation, next_s, real_action)
            updated_particles.append(next_s)
            if raw_weight > 0:
                weights.append(raw_weight)
            else:
                weights.append(self._sir_zero_weight_exponent)

        # Exaggerate weights (softmax)
        weights = list(scipy.special.softmax(np.array(weights)*temperature))

        # resample
        return np.random.choice(updated_particles,
                                num_sir_particles,
                                True,
                                weights).tolist()

    def clear_agent(self):
        self._agent = None  # forget about current agent so that can plan for another agent.
        self._last_num_sims = -1

    cpdef _search(self):
        cdef int sims_count = 0
        cdef double start_time, time_taken
        pbar = self._initialize_progress_bar()
        start_time = time.time()

        while not self._should_stop(sims_count, start_time):
            state = self._agent.sample_belief()
            if self._reseed_global_objective:
                self._global_objective = self._agent._policy_model.new_global_objective(state)
            self._global_objective = self._agent._policy_model.maintain_global_objective(state, self._global_objective)
            self._perform_simulation(state)
            sims_count += 1
            self._update_progress(pbar, sims_count, start_time)

        self._finalize_progress_bar(pbar)

        if not len(self._agent.tree.children):
            raise Exception("Root has no children.")

        best_action = self._agent.tree.argmax()

        print(self._agent.tree)
        print(self._agent.tree.children_data())
        print(f"best_action: {best_action}")
        print(f"sims_count: {sims_count}")

        time_taken = time.time() - start_time
        return best_action, time_taken, sims_count

    cdef _initialize_progress_bar(self):
        if self._show_progress:
            total = self._num_sims if self._num_sims > 0 else self._planning_time
            return tqdm(total=total)

    cpdef _perform_simulation(self, state):
        self._simulate(state, self._agent.history, self._agent.tree, None, None, 0)

    cdef bint _should_stop(self, int sims_count, double start_time):
        cdef float time_taken = time.time() - start_time
        if self._num_sims > 0:
            return sims_count >= self._num_sims
        else:
            return time_taken > self._planning_time

    cdef _update_progress(self, pbar, int sims_count, double start_time):
        if self._show_progress:
            pbar.n = sims_count if self._num_sims > 0 else round(time.time() - start_time, 2)
            pbar.refresh()

    cdef _finalize_progress_bar(self, pbar):
        if self._show_progress:
            pbar.close()

    cpdef Action _ucb(self, VNode root):
        """UCB1"""
        cdef Action best_action
        cdef float best_value
        best_action, best_value = None, float('-inf')
        for action in root.children:
            if root[action].num_visits == 0:
                val = float('inf')
            else:
                val = root[action].pref + \
                    self._exploration_const * np.sqrt(np.log(root.num_visits + 1) / root[action].num_visits)
            if val > best_value:
                best_action = action
                best_value = val
        return best_action

    # TODO: Add observation?
    cdef Action _sample_policy(self, State state, VNode root):
        # Option 1
        # if len(root.children) >= self._action_branching_factor:
        #     if np.random.random() > self._exploitation_prob:
        #         return root.least_visited()
        #     else:
        #         return root.sample_softmax(self._exploitation_temp, "pref")
        # else:
        #     self._global_objective = self._agent._policy_model.maintain_global_objective(state, self._global_objective)
        #     return self._agent._policy_model.sample(state, self._global_objective)

        # Option 2
        # cdef float r
        #
        # r = np.random.random()
        # if r <= self._exploitation_prob and len(root.children):
        #     # exploit
        #     # return root.least_visited()
        #     # if r > 0.5:
        #     #     return root.least_visited()
        #     # else:
        #     return root.sample_softmax(self._exploitation_temp, "pref")
        # else:
        #     self._global_objective = self._agent._policy_model.maintain_global_objective(state, self._global_objective)
        #     return self._agent._policy_model.sample(state, self._global_objective)

        # Option 3 (UCB)
        # TODO: rename _exploitation_prob to research tree?
        if np.random.random() <= self._exploitation_prob and len(root.children):
            return self._ucb(root)
        else:
            self._global_objective = self._agent._policy_model.maintain_global_objective(state, self._global_objective)
            return self._agent._policy_model.sample(state, self._global_objective)


    cpdef float _simulate(RGPOMCP self,
                          State state, tuple history, VNode root, QNode parent,
                          Observation observation, int depth):

        cdef int nsteps
        cdef Action action
        cdef State next_state
        cdef float reward
        cdef float r

        if root is None:
            if self._agent.tree is None:
                root = self._VNode(root=True)
                self._agent.tree = root
                if self._agent.tree.history != self._agent.history:
                    raise ValueError("Unable to plan for the given history.")
            else:
                root = self._VNode()

            if parent is not None:
                parent[observation] = root

        # Update belief and resample from root belief.
        if depth >= 1 and root is not None:
            root.belief.add(state)
        state = root.belief.random()
        action = self._sample_policy(state, root)

        # Sample generative model.
        next_state, observation, reward, nsteps = self._agent.blackbox_model.sample(state, action, self.discount_factor)

        if next_state.terminal:
            return reward

        if depth > self._max_depth:
            if hasattr(self._agent._policy_model, "value_heuristic"):
                return self._agent._policy_model.value_heuristic(state, next_state,
                                                       reward, self.discount_factor)
            else:
                return self._rollout(state, history, root, depth)

        # Create nodes if not created already.
        if root[action] is None:
            root[action] = QNode(pref=self._pref_init)

        if root[action][observation] is None:
            root[action][observation] = self._VNode(root=False)

        # Update counts and estimates
        # Option 1
        # cdef float d
        # d = root[action].num_visits * root[action].D - root[action][observation].num_visits * root[action][observation].V
        # root[action].num_visits += 1
        # root[action].R += (reward * self._reward_scale - root[action].R) / root[action].num_visits
        # root[action].D = (d + root[action][observation].num_visits \
        #                     * self._simulate(next_state,
        #                                       history + ((action, observation),),
        #                                       root[action][observation],
        #                                       root[action],
        #                                       observation,
        #                                       depth + nsteps)) / root[action].num_visits
        # root.num_visits += 1
        # root[action].pref = root[action].pref - root.V + root[action].R + root[action].D

        r = reward * self._reward_scale \
                          + self._simulate(next_state,
                                           history + ((action, observation),),
                                           root[action][observation],
                                           root[action],
                                           observation,
                                           depth + nsteps)

        # Option 2
        root.num_visits += 1
        root[action].num_visits += 1
        root[action].pref += (r - root[action].pref)/root[action].num_visits

        # root[action].pref = root[action].pref - root.V + root[action].R \
        #                     + (self.discount_factor ** nsteps) \
        #                     * self._simulate(next_state,
        #                                       history + ((action, observation),),
        #                                       root[action][observation],
        #                                       root[action],
        #                                       observation,
        #                                       depth + nsteps)

        # if depth == 0:
        #     print("Pref |", root[action].pref)

        # if depth == 0:
        #     print("V |", root.V)

        return r

    cpdef float _rollout(self, State state, tuple history, VNode root, int depth):
        cdef Action action
        cdef double discount = 1.0
        cdef double total_discounted_reward = 0
        cdef State next_state
        cdef Observation observation
        cdef double reward

        while depth < self._rollout_depth:
            if state.terminal:
                return total_discounted_reward
            action = self._agent._policy_model.rollout(state)
            next_state, observation, reward, nsteps = sample_generative_model(self._agent, state, action, self._discount_factor)
            history = history + ((action, observation),)
            depth += nsteps
            total_discounted_reward += reward * self._reward_scale * discount
            discount *= (self._discount_factor ** nsteps)
            state = next_state

        return total_discounted_reward

    def _VNode(self, root=False, **kwargs):
        """Returns a VNode with default values; The function naming makes it clear
        that this function is about creating a VNode object."""
        if root:
            return RootVNodeParticles(num_visits=0,
                             history=self._agent.history,
                             belief=copy.deepcopy(self._agent.belief))

        else:
            return VNodeParticles(num_visits=0,
                                  belief=Particles([]))