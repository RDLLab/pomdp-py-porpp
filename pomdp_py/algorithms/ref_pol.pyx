from dis import disco

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

    def __init__(self, num_visits=0):
        super().__init__()
        self.num_visits = num_visits
        self.children = {}

    def __str__(self):
        return typ.red(f'QNode<N:{self.num_visits} | {self.children.keys()}>')

    def __repr__(self):
        return self.__str__()

cdef class VNode(TreeNode):
    """A history node in the search tree."""
    def __init__(self, num_visits=0):
        super().__init__()
        self.num_visits = num_visits
        self.children = {}

    def __str__(self):
        return f'VNode<N:{self.num_visits} | num_children: {len(self.children)}>'

    def __repr__(self):
        return self.__str__()

cdef class VNodeParticles(VNode):
    """A history node with a particle representation of the associated belief."""

    def __init__(self, num_visits=0, belief=Particles([])):
        self.num_visits = num_visits
        self.belief = belief
        self.children = {}

    def __str__(self):
        return f'VNode<N:{self.num_visits}, Num Particles:{len(self.belief)} | num_children: {len(self.children)}>'

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

cdef class RefPol(Planner):

    """RefPol executes a reference policy as an open-loop policy without planning for
    transition and observation uncertainty.

    Args:
        discount_factor (float): The single-step discount factor. Default: 0.99.
        num_sir_particles (int): The number of particles to maintain as part of SIR belief update. Default: 1000.
        sir_temp (int): A positive real number indicating the temperature of the softmax used to exaggerate
            resampling weights. Default: 1.
        sir_zero_weight_exponent (float): A negative float x for which exp(x) re-assigns the weight for resampling a
            particle that has an effective resampling weight of zero. Default: -Inf.
        reseed_global_objective (bool): Reseed global objective after every simulation. Default: True.
    """

    def __init__(self,
                 discount_factor=0.99,
                 num_sir_particles=1000,
                 sir_temp=1.,
                 sir_zero_weight_exponent=-np.inf,
                 reseed_global_objective=True):

        self._discount_factor = discount_factor
        self._num_sir_particles = num_sir_particles
        self._sir_temp = sir_temp
        self._sir_zero_weight_exponent = sir_zero_weight_exponent

        self._reseed_global_objective = reseed_global_objective
        self._global_objective = None

        # to simplify function calls; plan only for one agent at a time
        self._agent = None

    def __str__(self):
        return f'RefPol Params[' \
               f'discount_factor:{self._discount_factor}, ' \
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
    def discount_factor(self):
        return self._discount_factor

    @property
    def global_objective(self):
        return self._global_objective

    cpdef public plan(self, Agent agent):
        cdef Action action

        self._agent = agent   # switch focus on planning for the given agent
        if not hasattr(self._agent, "tree"):
            self._agent.add_attr("tree", None)

        state = self._agent.sample_belief()
        if self._reseed_global_objective:
            self._global_objective = self._agent._policy_model.new_global_objective(state)
        self._global_objective = self._agent._policy_model.maintain_global_objective(state, self._global_objective)

        action = self._sample_policy(state, self._agent.tree)
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
                            "PORPI not usable. Please convert it to particles.")

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

    # TODO: Add observation?
    cdef Action _sample_policy(self, State state, VNode root):

        cdef Action action

        if root is None:
            if self._agent.tree is None:
                root = self._VNode(root=True)
                self._agent.tree = root
                if self._agent.tree.history != self._agent.history:
                    raise ValueError("Unable to plan for the given history.")
            else:
                root = self._VNode()

        self._global_objective = self._agent._policy_model.maintain_global_objective(state, self._global_objective)
        action = self._agent._policy_model.sample(state, self._global_objective)

        if root[action] is None:
            root[action] = QNode()

        return action

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