from pomdp_py.framework.planner cimport Planner
from pomdp_py.framework.basics cimport Agent, Objective, Action, State, Observation
from pomdp_py.representations.distribution.particles cimport Particles

cdef class TreeNode:
    cdef public dict children
    cdef public int num_visits

cdef class QNode(TreeNode):
    pass

cdef class VNode(TreeNode):
    pass

cdef class RootVNode(VNode):
    cdef public tuple history

cdef class VNodeParticles(VNode):
    cdef public Particles belief
cdef class RootVNodeParticles(RootVNode):
    cdef public Particles belief

cdef class RefPol(Planner):
    cdef float _discount_factor
    cdef int _num_sir_particles
    cdef float _sir_temp
    cdef float _sir_zero_weight_exponent

    cdef bint _reseed_global_objective
    cdef Objective _global_objective

    cdef Agent _agent
    cdef list _sir(self, Agent agent, Action real_action, Observation real_observation,
                   list particles, float temperature, int num_sir_particles)

    cdef Action _sample_policy(self, State state, VNode root)