from pomdp_py.algorithms.po_uct cimport VNode, RootVNode, POUCT
from pomdp_py.framework.basics cimport Agent, Action, Observation
from pomdp_py.representations.distribution.particles cimport Particles

cdef class VNodeParticles(VNode):
    cdef public Particles belief
cdef class RootVNodeParticles(RootVNode):
    cdef public Particles belief

cdef class POMCP(POUCT):

    cdef int _num_sir_particles
    cdef float _sir_temp
    cdef float _sir_zero_weight_exponent

    cdef list _sir(POMCP self, Agent agent, Action real_action, Observation real_observation,
                   list particles, float temperature, int num_sir_particles)