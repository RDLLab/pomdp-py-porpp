from pomdp_py.framework.planner cimport Planner
from pomdp_py.framework.basics cimport Agent, Action, State, Observation, Objective
from pomdp_py.representations.distribution.particles cimport Particles

cdef class TreeNode:
    cdef public dict children
    cdef public int num_visits

cdef class QNode(TreeNode):
    cdef public double R
    cdef public double D

cdef class VNode(TreeNode):
    cdef public double V
    cdef public double W
    cdef public dict policy

cdef class RootVNode(VNode):
    cdef public tuple history

cdef class VNodeParticles(VNode):
    cdef public Particles belief

cdef class RootVNodeParticles(RootVNode):
    cdef public Particles belief

cdef class RefSolver(Planner):
    cdef float _planning_time
    cdef int _num_sims
    cdef int _max_depth
    cdef int _rollout_depth
    cdef float _discount_factor
    cdef float _temperature
    cdef float _exploration_const
    cdef int _num_visits_init
    cdef double _value_init
    cdef int _num_sir_particles
    cdef float _sir_temp
    cdef double _sir_zero_weight_exponent

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
    cpdef update_policy(self, VNode root)
    cpdef Action _sample_policy(self, State state, VNode root)
    cpdef _simulate(self, State state, tuple history, VNode root, QNode parent,
                    Observation observation, int depth)
    cpdef _maintain_expectation(self, VNode root, Action action, Observation observation,
                                State next_state, float reward,
                                tuple history, int depth, int nsteps)
    cpdef _rollout(self, State state, tuple history, VNode root, int depth)