from pomdp_py.framework.planner cimport Planner
from pomdp_py.framework.basics cimport Agent, Objective, Action, State, Observation
from pomdp_py.representations.distribution.particles cimport Particles

cdef class TreeNode:
    cdef public dict children
    cdef public int num_visits

cdef class QNode(TreeNode):
    cdef public float R
    cdef public float D
    cdef public float pref

cdef class VNode(TreeNode):
    cdef public float V
    cdef public float W
    cdef public float pref_shift

    cpdef least_visited(VNode self)

    cpdef argmax(VNode self)

    cpdef mellowmax(VNode self, float temperature)

    cpdef boltzmann(VNode self, float temperature)

    cpdef sample_softmax(VNode self, float temperature, str value)

cdef class RootVNode(VNode):
    cdef public tuple history

cdef class VNodeParticles(VNode):
    cdef public Particles belief
cdef class RootVNodeParticles(RootVNode):
    cdef public Particles belief

cdef class PORPI(Planner):
    cdef float _planning_time
    cdef int _num_sims
    cdef int _max_depth
    cdef int _rollout_depth
    cdef float _discount_factor
    cdef float _temperature
    cdef float _pref_init
    cdef int _action_branching_factor
    cdef float _exploitation_prob
    cdef float _exploitation_temp
    cdef int _num_sir_particles
    cdef float _sir_temp
    cdef float _sir_zero_weight_exponent
    cdef float _reward_scale

    cdef bint _reseed_global_objective
    cdef Objective _global_objective

    cdef bint _show_progress
    cdef int _pbar_update_interval

    cdef Agent _agent
    cdef int _last_num_sims
    cdef float _last_planning_time
    cdef list _sir(self, Agent agent, Action real_action, Observation real_observation,
                   list particles, float temperature, int num_sir_particles)
    cpdef _search(self)
    cdef _initialize_progress_bar(self)
    cpdef _perform_simulation(self, State state)
    cdef bint _should_stop(self, int sims_count, double start_time)
    cdef _update_progress(self, pbar, int sims_count, double start_time)
    cdef _finalize_progress_bar(self, pbar)
    cdef Action _sample_policy(self, State state, VNode root)
    cpdef float _simulate(PORPI self,
                    State state, tuple history, VNode root, QNode parent,
                    Observation observation, int depth)
    cpdef float _rollout(self, State state, tuple history, VNode root, int depth)