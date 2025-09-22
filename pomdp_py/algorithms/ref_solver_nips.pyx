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
    """An action node of the belief tree."""
    def __init__(self, num_visits=0, R=0.0):
        super().__init__()
        self.R = R
        self.num_visits = num_visits

    def __str__(self):
        return typ.red(f'QNode<N:{self.num_visits}, R:{self.R} | {(self.children.keys())}>')

    def __repr__(self):
        return self.__str__()

cdef class VNode(TreeNode):
    """A belief node of the belief tree."""
    def __init__(self, num_visits=0, Z=1.0, G=0.0):
        super().__init__()
        self.num_visits = num_visits
        self.Z = Z
        self.G = G
        self.policy = dict()

    def __str__(self):
        return f'VNode<N:{self.num_visits}, Z:{self.Z}, G:{self.G} | num_children: {len(self.children)}>'

    def __repr__(self):
        return self.__str__()

    def children_data(self):
        string = ""
        for action in self.children:
            string += (f' |_{action}, N:{self[action].num_visits}, '
                       f'R:{round(self[action].R, 2)}\n')
        return string

cdef class VNodeParticles(VNode):
    def __init__(self, num_visits=0, belief=Particles([])):
        super().__init__(num_visits)
        self.belief = belief

    def __str__(self):
        return f'VNode<N:{self.num_visits}, V:{self.V}, Num Particles:{len(self.belief)} | {(self.children.keys())}>'

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

cdef class RefSolverNIPS(Planner):

    """
    The RefSolver used in Kim et al. (NeurIPS2023) "Reference-Based POMDPs".

    Args:
        planning_time (float): The amount of time given to each planning step (seconds). Default: -1.
            if negative, then planning terminates when number of simulations `num_sims` reached.
            If both `num_sims` and `planning_time` are negative, then the planner will test_utils for 1 second.
        num_sims (int): Number of simulations for each planning step. If negative,
            then will terminate when planning_time is reached.
            If both `num_sims` and `planning_time` are negative, then the planner will test_utils for 1 second.
        max_depth (int): The maximum depth of the search tree (Default: 30).
        rollout_depth (int): The maximum depth of a rollout (Default: 50)
        discount_factor (float): The discount factor (Default: .99)
        reward_scaling_factor (float): The reward scaling factor (Default: 1.0)
        num_visits_init (int): The default number of initial visits for a new node (Default: 1)
        value_init (float): The default node value (Default: 0.)
        num_sir_particles (int): The number of particles to maintain as part of SIR belief update. Default: 1000.
        sir_temp (int): A positive real number indicating the temperature of the softmax used to exaggerate
            resampling weights. Default: 1.
        sir_zero_weight_exponent (float): A negative float x for which exp(x) re-assigns the weight for resampling a
            particle that has an effective resampling weight of zero. Default: -Inf.
        reseed_global_objective (bool): Reseed global objective after every simulation. Default: True.
        show_progress (bool): True if print a progress bar for simulations.
        pbar_update_interval (int): The number of simulations to test_utils after each update of the progress bar,
            Only useful if show_progress is True; You can set this parameter even if your stopping criteria
            is time.
    """
    def __init__(self,
                 planning_time=1.0, num_sims=-1,
                 max_depth=30, rollout_depth=50,
                 discount_factor=0.99,
                 reward_scaling_factor=1.0,
                 num_visits_init=1, value_init=0.,
                 num_sir_particles=1000, sir_temp=0.01,
                 sir_zero_weight_exponent=-np.inf,
                 reseed_global_objective=True,
                 show_progress=False, pbar_update_interval=5):
        self._planning_time = planning_time
        self._num_sims = num_sims
        if self._num_sims < 0 and self._planning_time < 0:
            self._planning_time = 1.
        self._max_depth = max_depth
        self._rollout_depth = rollout_depth
        self._discount_factor = discount_factor
        self._reward_scaling_factor = reward_scaling_factor
        self._num_visits_init = num_visits_init
        self._value_init = value_init
        self._num_sir_particles = num_sir_particles
        self._sir_temp = sir_temp
        self._sir_zero_weight_exponent = sir_zero_weight_exponent

        self._reseed_global_objective = reseed_global_objective
        self._global_objective = None

        self._show_progress = show_progress
        self._pbar_update_interval = pbar_update_interval

        # to simplify function calls; plan only for one agent at a time
        self._agent = None
        self._last_num_sims = -1
        self._last_planning_time = -1

    def __str__(self):
        return f'RefSolverNIPS Params[' \
               f'planning_time:{self._planning_time:.3f}, ' \
               f'num_sims:{self._num_sims}, ' \
               f'max_depth:{self._max_depth}, ' \
               f'rollout_depth:{self._rollout_depth}, ' \
               f'discount_factor:{self._discount_factor:.3f}, ' \
               f'reward_scaling_factor:{self._reward_scaling_factor:.3f}, ' \
               f'num_visits_init:{self._num_visits_init:.3f}, ' \
               f'value_init:{self._value_init:.3f}, ' \
               f'num_sir_particles:{self._num_sir_particles}, ' \
               f'sir_temp:{self._sir_temp}, ' \
               f'sir_zero_weight_exponent:{self._sir_zero_weight_exponent}, '\
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
        return self._last_num_sims

    @property
    def last_planning_time(self):
        """Returns the amount of time (seconds) ran for the last `plan` call."""
        return self._last_planning_time

    @property
    def max_depth(self):
        return self._max_depth

    @property
    def rollout_depth(self):
        return self._rollout_depth

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

        self._agent = agent  # switch focus on planning for the given agent
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
                            "RefSolver not usable. Please convert it to particles.")

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
                raise (ValueError, 'Action was a MacroAction. Expecting a MacroObservation too.')

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

        self.update_policy(self._agent.tree)
        if not len(self._agent.tree.policy):
            print("Warning: policy at the root tree is empty")
            raise Exception("Empty policy")

        best_action = max(self._agent.tree.policy, key=self._agent.tree.policy.get)

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

    cpdef _perform_simulation(self, State state):
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

    cpdef update_policy(self, VNode root):
        cdef dict uu_opt_raw = {}  # will store the optimal "belief-to-belief" transition probability.
        cdef float uu_normaliser = 1e-5
        cdef dict uu_opt = {};

        for action in root.children:
            for observation in root[action].children:
                # print("r: ", root[action].R)
                # print("z: ", root[action][observation].Z)
                # print(root.num_visits)
                x = root[action][observation].num_visits \
                * root[action][observation].Z \
                * np.exp(root[action].R) / root.num_visits
                # print("x: ", x)
                uu_opt_raw[action, observation] = x
                # print("UU: ", uu_opt_raw[action,observation])
                uu_normaliser += x

        for (ao, uu_weight) in uu_opt_raw.items():
            uu_opt[ao] = uu_weight / uu_normaliser

        # uu_opt = {ao: uu_weight / uu_normaliser for (ao, uu_weight) in uu_opt_raw.items()}
        cdef dict u_opt = dict()
        cdef float u_normaliser = 1e-5
        cdef float pi_a = 0.
        for action in root.children:
            if root[action] is None or root[action].num_visits == 0:
                u_opt[action] = 1e-5
            else:
                ha = root[action]
                try:
                    if uu_opt[action, observation] == 0:
                        pi_a = 0.
                    else:
                        for observation in ha.children:
                            pi_a = ha[observation].num_visits / ha.num_visits * np.log(ha[observation].num_visits / ha.num_visits / uu_opt[action, observation])
                except:
                    print("Update pi_a failed...")
                    pi_a = 0.

                u_opt[action] = np.exp(-pi_a)

            u_normaliser += u_opt[action]

        for (a, p) in u_opt.items():
            prob = p / u_normaliser
            root.policy[a] = prob
        return

    cpdef Action _sample_policy(self, State state, root: VNode):

        self._global_objective = self._agent._policy_model.maintain_global_objective(state, self._global_objective)

        return self._agent._policy_model.sample(state, self._global_objective)

    cpdef _simulate(self, State state, tuple history, VNode root, QNode parent, Observation observation, int depth):
        cdef int nsteps
        cdef Action action
        cdef State next_state
        cdef double reward

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
        root.num_visits += 1
        state = root.belief.random()
        action = self._sample_policy(state, root)

        # Sample generative model.
        next_state, observation, reward, nsteps = self._agent.blackbox_model.sample(state, action, self.discount_factor)

        reward *= self._reward_scaling_factor

        if depth > self._max_depth:
            if hasattr(self._agent._policy_model, "value_heuristic"):
                return self._agent._policy_model.value_heuristic(state, next_state,
                                                       reward, self.discount_factor)
            else:
                return self._rollout(state, history, root, depth)

        action = self._sample_policy(state, root)

        # Create nodes if not created already.
        if root[action] is None:
            root[action] = QNode(self._num_visits_init, self._value_init)

        if root[action][observation] is None:
            root[action][observation] = self._VNode(root=False)

        # Update estimates.
        root[action].num_visits += 1
        root[action].R += (reward - root[action].R) / root[action].num_visits

        """
        WARNING! (Edward Kim: 17 Mar 2024)
        This is an error in the original paper, since the early additions to the z_value use old versions of the 
        expected reward (i.e. root[a].R) whereas later estimates use the most up-to-date ones. The early summands
        are not updated to use the latest one which can introduce biases to the sum. The error is intentional preserved
        in this code.
        """
        root.Z = root.Z + (np.exp(root[action].R) *
                           self._simulate(next_state,
                                          history + ((action, observation),),
                                          root[action][observation],
                                          root[action],
                                          observation, depth + nsteps) - root.Z) / root.num_visits
        return root.Z ** (self._discount_factor ** nsteps)

    cpdef _rollout(self, State state, tuple history, VNode root, int depth):
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
            total_discounted_reward += reward * discount * self._reward_scaling_factor
            discount *= (self._discount_factor ** nsteps)
            state = next_state

        return total_discounted_reward

    def _VNode(self, root=False, **kwargs):
        """Returns a VNode with default values; The function naming makes it clear
        that this function is about creating a VNode object."""
        if root:
            return RootVNodeParticles(self._num_visits_init,
                                      self._agent.history,
                                      belief=copy.deepcopy(self._agent.belief))

        else:
            return VNodeParticles(self._num_visits_init,
                                  belief=Particles([]))
