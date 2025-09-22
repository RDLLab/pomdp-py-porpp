"""
We implement POMCP as described in the original paper
Monte-Carlo Planning in Large POMDPs
https://papers.nips.cc/paper/4031-monte-carlo-planning-in-large-pomdps

One thing to note is that, in this algorithm, belief
update happens as the simulation progresses. The new
belief is stored in the vnodes at the level after
executing the next action. These particles will
be reinvigorated if they are not enough.

However, it is possible to separate MCTS completely
from the belief update. This means the belief nodes
no longer keep track of particles, and belief update
and particle reinvogration happen for once after MCTS
is completed. I have previously implemented this version.
This version is also implemented in BasicPOMCP.jl
(https://github.com/JuliaPOMDP/BasicPOMCP.jl)
The two should be EQUIVALENT. In general, it doesn't
hurt to do the belief update during MCTS, a feature
of using particle representation.
"""

from pomdp_py.framework.basics cimport *
from pomdp_py.framework.planner cimport Planner
from pomdp_py.representations.distribution.particles cimport Particles
from pomdp_py.representations.belief.particles cimport particle_reinvigoration
from pomdp_py.algorithms.po_uct cimport VNode, RootVNode, QNode, POUCT, RandomRollout
import copy
import scipy
import numpy as np
import math
import time

cdef class VNodeParticles(VNode):
    """POMCP's VNode maintains particle belief"""
    def __init__(self, num_visits, belief=Particles([])):
        self.num_visits = num_visits
        self.belief = belief
        self.children = {}  # a -> QNode
    def __str__(self):
        return "VNode(%.3f, %.3f, %d | %s)" % (self.num_visits, self.value, len(self.belief),
                                               str(self.children.keys()))
    def __repr__(self):
        return self.__str__()

cdef class RootVNodeParticles(RootVNode):
    def __init__(self, num_visits, history, belief=Particles([])):
        # vnodeobj = VNodeParticles(num_visits, value, belief=belief)
        RootVNode.__init__(self, num_visits, history)
        self.belief = belief
    @classmethod
    def from_vnode(cls, vnode, history):
        rootnode = RootVNodeParticles(vnode.num_visits, history, belief=vnode.belief)
        rootnode.children = vnode.children
        return rootnode

cdef class POMCP(POUCT):

    """POMCP is POUCT + particle belief representation.
    This POMCP version only works for problems
    with action space that can be enumerated."""

    def __init__(self,
                 max_depth=5,
                 planning_time=-1., num_sims=-1,
                 discount_factor=0.9, exploration_const=math.sqrt(2),
                 num_visits_init=0, value_init=0,
                 num_sir_particles=1000, sir_temp=.01, sir_zero_weight_exponent=-np.inf, # Custom fields added for resampling.
                 rollout_policy=RandomRollout(), action_prior=None,
                 show_progress=False, pbar_update_interval=5):
        super().__init__(max_depth=max_depth,
                         planning_time=planning_time,
                         num_sims=num_sims,
                         discount_factor=discount_factor,
                         exploration_const=exploration_const,
                         num_visits_init=num_visits_init,
                         value_init=value_init,
                         rollout_policy=rollout_policy,
                         action_prior=action_prior,
                         show_progress=show_progress,
                         pbar_update_interval=pbar_update_interval)

        # Customized from original PO-UCT in pomdp_py framework.
        self._num_sir_particles = num_sir_particles
        self._sir_temp = sir_temp
        self._sir_zero_weight_exponent = sir_zero_weight_exponent

    def __str__(self):
        return f'POMCP Params[' \
               f'planning_time:{self._planning_time:.3f}, ' \
               f'num_sims:{self._num_sims}, ' \
               f'max_depth:{self._max_depth}, ' \
               f'discount_factor:{self._discount_factor:.3f}, ' \
               f'exploration_const:{self._exploration_const:.3f}, ' \
               f'num_visits_init:{self._num_visits_init}, ' \
               f'value_init:{self._value_init:.3f}, ' \
               f'rollout_policy:{self._rollout_policy}, ' \
               f'num_sir_particles:{self._num_sir_particles}, ' \
               f'sir_temp:{self._sir_temp}, ' \
               f'sir_zero_weight_exponent:{self._sir_zero_weight_exponent}]'

    @property
    def update_agent_belief(self):
        """True if planner's update function also updates agent's
        belief."""
        return True

    @property
    def discount_factor(self):
        return self._discount_factor

    def plan(self, agent):
        cdef Action action
        cdef float time_taken
        cdef int sims_count

        # Only works if the agent's belief is particles
        if not isinstance(agent.belief, Particles):
            raise TypeError("Agent's belief is not represented in particles.\n"\
                            "POMCP not usable. Please convert it to particles.")

        return POUCT.plan(self, agent)

    # Original update function -- does not use resampling...
    #---------------------------------------------------------------------------------#
    # cpdef update(self, Agent agent, Action real_action, Observation real_observation,
    #              state_transform_func=None):
    #     """
    #     Assume that the agent's history has been updated after taking real_action
    #     and receiving real_observation.
    #
    #     `state_transform_func`: Used to add artificial transform to states during
    #         particle reinvigoration. Signature: s -> s_transformed
    #     """
    #     if not isinstance(agent.belief, Particles):
    #         raise TypeError("agent's belief is not represented in particles.\n"\
    #                         "POMCP not usable. Please convert it to particles.")
    #     if not hasattr(agent, "tree"):
    #         print("Warning: agent does not have tree. Have you planned yet?")
    #         return
    #
    #     if agent.tree[real_action][real_observation] is None:
    #         # Never anticipated the real_observation. No reinvigoration can happen.
    #         raise ValueError("Particle deprivation.")
    #     # Update the tree; Reinvigorate the tree's belief and use it
    #     # as the updated belief for the agent.
    #     agent.tree = RootVNodeParticles.from_vnode(agent.tree[real_action][real_observation],
    #                                                agent.history)
    #     tree_belief = agent.tree.belief
    #     agent.set_belief(particle_reinvigoration(tree_belief,
    #                                              len(agent.init_belief.particles),
    #                                              state_transform_func=state_transform_func))
    #     # If observation was never encountered in simulation, then tree will be None;
    #     # particle reinvigoration will occur.
    #     if agent.tree is not None:
    #         agent.tree.belief = copy.deepcopy(agent.belief)
    #---------------------------------------------------------------------------------#

    # cpdef update(self, Agent agent, Action real_action, Observation real_observation,
    #              state_transform_func=None):
    #     """
    #     Use sampling importance resampling to update the belief to reduce the likelihood
    #     of particle deprivation. Particle reinvigoration is used to ensure a rich set of
    #     particles. For technical details refer to the textbook by (Thrun, 2002).
    #
    #     Assumes self._agent.observation_model.probability(o, next_s, a) is implemented.
    #     """
    #     cdef list particles
    #     cdef list existing_particles
    #     cdef int num_new_particles
    #     cdef RootVNodeParticles vnode
    #
    #     if not isinstance(agent.belief, Particles):
    #         raise TypeError("Agent's belief is not represented by particles.\n"\
    #                         "POMCP not usable. Please convert it to particles.")
    #
    #     if not hasattr(agent, "tree") or agent.tree is None:
    #         print("Warning: agent does not have tree. Have you planned yet?")
    #         return
    #
    #     # Always perform resampling. This is because the continuous action space
    #     # means that the executed action may not be in the tree for very sparsely
    #     # separated beliefs.
    #
    #     particles = agent.tree.belief.particles
    #
    #     if isinstance(real_action, MacroAction):
    #         if not isinstance(real_observation, MacroObservation):
    #             raise (ValueError, 'Action was a MacroAction. Expecting a MacroObservation too.')
    #
    #         # sampling importance resampling
    #         for a, o in zip(real_action.action_sequence, real_observation.observation_sequence):
    #             particles = self._sir(agent, a, o, particles,
    #                                   self._sir_temp, self._num_sir_particles)
    #     else:
    #         particles = self._sir(agent, real_action, real_observation, particles,
    #                               self._sir_temp, self._num_sir_particles)
    #
    #     # Update the tree and use the resampled belief.
    #     vnode = RootVNodeParticles.from_vnode(VNodeParticles(num_visits=self._num_visits_init, belief=Particles(copy.deepcopy(particles))),
    #                                                agent.history)
    #
    #     agent.set_belief(vnode.belief)
    #
    #     if real_action not in agent.tree or real_observation not in agent.tree[real_action]:
    #         agent.tree = None
    #     elif agent.tree[real_action][real_observation] is not None:
    #         # Update the tree (prune)
    #         agent.tree = RootVNodeParticles.from_vnode(
    #             agent.tree[real_action][real_observation],
    #             agent.history)
    #     else:
    #         raise ValueError("Unexpected state; child should not be None")

    cpdef update(self, Agent agent, Action real_action, Observation real_observation,
                 state_transform_func=None):
        """
        Use sampling importance resampling to update the belief to reduce the likelihood 
        of particle deprivation. Particle reinvigoration is used to ensure a rich set of 
        particles. For technical details refer to the textbook by (Thrun, 2002).

        Assumes self._agent.observation_model.probability(o, next_s, a) is implemented.
        """
        cdef list particles

        if not isinstance(agent.belief, Particles):
            raise TypeError("Agent's belief is not represented by particles.\n"\
                            "Planner is not usable. Please convert it to particles.")

        if not hasattr(agent, "tree") or agent.tree is None:
            print("Warning: agent does not have tree. Have you planned yet?")
            return

        # Always perform resampling. This is because the continuous action space
        # means that the executed action may not be in the tree for very sparsely
        # separated beliefs.

        particles = agent.tree.belief.particles

        if isinstance(real_action, MacroAction):
            if not isinstance(real_observation, MacroObservation):
                raise ValueError('Action was a MacroAction. Expecting a MacroObservation too.')

            # sampling importance resampling
            for a, o in zip(real_action.action_sequence, real_observation.observation_sequence):
                particles = self._sir(agent, a, o, particles,
                                      self._sir_temp, self._num_sir_particles)
        else:
            particles = self._sir(agent, real_action, real_observation, particles,
                                  self._sir_temp, self._num_sir_particles)

        # Update the tree and use the resampled belief.
        vnode = RootVNodeParticles.from_vnode(VNodeParticles(num_visits=self._num_visits_init, belief=Particles(copy.deepcopy(particles))),
                                                   agent.history)

        agent.set_belief(vnode.belief)

        if real_action not in agent.tree or real_observation not in agent.tree[real_action]:
            agent.tree = None
        elif agent.tree[real_action][real_observation] is not None:
            # Update the tree (prune)
            agent.tree = RootVNodeParticles.from_vnode(
                agent.tree[real_action][real_observation],
                agent.history)
        else:
            raise ValueError("Unexpected state; child should not be None")

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


    # cpdef _simulate(POMCP self,
    #                 State state, tuple history, VNode root, QNode parent,
    #                 Observation observation, int depth):
    #     total_reward = POUCT._simulate(self, state, history, root, parent, observation, depth)
    #     if depth == 1 and root is not None:
    #         root.belief.add(state)  # belief update happens as simulation goes.
    #     return total_reward

    # Rewritten to accommodate resampling and value heuristics.
    # Note the previous implementation didn't continually add new particles to child nodes.
    cpdef _simulate(POMCP self,
                    State state, tuple history, VNode root, QNode parent,
                    Observation observation, int depth):

        if depth > self._max_depth:
            return 0.

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

            self._expand_vnode(root, history, state=state)
            if hasattr(self._agent._policy_model, "value_heuristic"):
                return self._agent._policy_model.value_heuristic(state, None,
                                                       0, self.discount_factor)
            else:
                return self._rollout(state, history, root, depth)

        cdef int nsteps
        action = self._ucb(root)
        # next_state, observation, reward, nsteps = sample_generative_model(self._agent, state, action)
        next_state, observation, reward, nsteps = self._agent.blackbox_model.sample(state, action, self.discount_factor)
        if nsteps == 0:
            # This indicates the provided action didn't lead to transition
            # Perhaps the action is not allowed to be performed for the given state
            # (for example, the state is not in the initiation set of the option,
            # or the state is a terminal state)
            return reward

        total_reward = reward + (self._discount_factor ** nsteps) * self._simulate(next_state,
                                                                                   history + ((action, observation),),
                                                                                   root[action][observation],
                                                                                   root[action],
                                                                                   observation,
                                                                                   depth + nsteps)

        if depth >= 1 and root is not None:
            root.belief.add(state)
        root.num_visits += 1
        root[action].num_visits += 1
        root[action].value = root[action].value + (total_reward - root[action].value) / (root[action].num_visits)
        return total_reward

    def _VNode(self, root=False, **kwargs):
        """Returns a VNode with default values; The function naming makes it clear
        that this function is about creating a VNode object."""
        if root:
            # agent cannot be None.
            return RootVNodeParticles(self._num_visits_init,
                                      self._agent.history,
                                      belief=copy.deepcopy(self._agent.belief))
        else:
            return VNodeParticles(self._num_visits_init,
                                  belief=Particles([]))
