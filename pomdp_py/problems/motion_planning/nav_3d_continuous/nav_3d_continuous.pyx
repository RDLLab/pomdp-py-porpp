"""This file provides all plugins for the POMDP model.
"""

from pomdp_py.framework.basics cimport *
from pomdp_py.algorithms.po_uct cimport *
from pomdp_py.representations.belief.particles cimport *
from pomdp_py.problems.motion_planning.environments cimport *
from pomdp_py.problems.motion_planning.environments.pyb_env cimport PyBulletEnv

import pomdp_py
from pomdp_py.problems.motion_planning.path_planning.prm import PRM
import numpy as np
import pybullet as pyb
import time
import pathlib

np.set_printoptions(precision=3, suppress=True)  # for neat printing of numpy arrays.

cdef angle_distance(float angle1, float angle2):

    return min(2*np.pi - abs(angle1 - angle2), abs(angle1 - angle2))

cdef class Nav3DState(State):
    EPSILON = 1.0
    cdef public tuple _position
    cdef public bint _terminal
    cdef public bint _danger_zone
    cdef public bint _landmark
    cdef public bint _goal
    def __init__(self, position, danger_zone, landmark, goal):
        """
        position (numpy array): The state position.
        danger_zone (bool): The robot is at a danger zone.
        landmark (bool): The robot is at a landmark.
        goal (bool): The robot is at the goal.

        """
        self._position = position
        self._terminal = danger_zone or goal
        self._danger_zone = danger_zone
        self._landmark = landmark
        self._goal = goal
        # self._hash = hash((self._position, self._terminal, self._danger_zone, self._landmark, self._goal))

    def __hash__(self):
        # Keep this set to 1 if you want to identify states via __eq__.
        # Python dictionaries identify keys if (__hash__ and __eq__) is True.
        return 1
        # return self._hash

    def __eq__(self, other):
        if isinstance(other, Nav3DState):
            return (np.linalg.norm(np.array(self._position) - np.array(other._position), ord=2) < Nav3DState.EPSILON and
                    self._terminal == other._terminal and
                    self._danger_zone == other._danger_zone and
                    self._landmark == other._landmark and
                    self._goal == other._goal)
        else:
            return False

    def __str__(self):
        return (f"<pos: ({self._position[0]:.3f}, {self._position[1]:.3f}, {self._position[2]:.3f}) "
                f"| dz: {self._danger_zone} | lm: {self._landmark} | goal: {self._goal}>")

    def __repr__(self):
        return self.__str__()

    @property
    def xyz(self):
        return self._position

    @property
    def terminal(self):
        return self._terminal

    @property
    def is_goal(self):
        return self._goal

cdef class Nav3DAction(Action):

    EPSILON_LONGI = np.pi/6
    EPSILON_LATI = np.pi/6
    cdef public float _longitude
    cdef public float _latitude

    def __init__(self, longitude, latitude):

        if longitude > np.pi or longitude < -np.pi:
            raise(ValueError, "Make sure longitude is in radians and in the principal range.")

        if latitude > np.pi/2 or latitude < -np.pi/2:
            raise(ValueError, "Make sure latitude is in radians and in [-pi/2, pi/2].")

        self._longitude = longitude
        self._latitude = latitude

    def __hash__(self):
        # Keep this set to 1 if you want to identify actions via __eq__.
        # Python dictionaries identify keys if (__hash__ and __eq__) is True.
        return 1
        # return self._hash

    def __eq__(self, other):
        if isinstance(other, Action):
            return (angle_distance(self._latitude, other._latitude) <= self.EPSILON_LATI
                    and angle_distance(self._longitude, other._longitude) <= self.EPSILON_LONGI)
        return False

    def __str__(self):
        return f"<longitude (rad): {self._longitude:.3f}, latitude (rad): {self._latitude:.3f}>"

    def __repr__(self):
        return self.__str__()


cdef class Nav3DMacroAction(MacroAction):

    EPSILON_ANGLE = np.pi/3
    EPSILON_EFFECT = 5

    def __init__(self, action_sequence):
        """
        action_sequence (list): A list of primitive actions
        """

        self._action_sequence = action_sequence
        # self._hash = hash(tuple(action_sequence))

    def to_vector(self, Nav3DAction action):
        longitude, latitude = action._longitude, action._latitude
        return np.array([np.cos(longitude) * np.cos(latitude), np.sin(longitude) * np.cos(latitude), np.sin(latitude)])

    def effect(self):
        vectors = list(map(self.to_vector, self._action_sequence))
        return sum(vectors)

    def angle_between(self, v1, v2):
        return np.arccos(np.clip(np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2), -1.0, 1.0))

    def __hash__(self):
        # Keep this set to 1 if you want to identify actions via __eq__.
        # Python dictionaries identify keys if (__hash__ and __eq__) is True.
        return 1
        # return self._hash

    def __eq__(self, Nav3DMacroAction other):
        if isinstance(other, Nav3DMacroAction):
            return self._action_sequence == other._action_sequence
        return False

    # def __eq__(self, Nav3DMacroAction other):
    #     if isinstance(other, Nav3DMacroAction):
    #         # if np.linalg.norm(self.effect() - other.effect()) < self.EPSILON * min(len(self._action_sequence), len(other._action_sequence)):
    #         if self.angle_between(self.effect(), other.effect()) < self.EPSILON_ANGLE and np.linalg.norm(self.effect() - other.effect()) <= self.EPSILON_EFFECT:
    #             return True
    #         return False

    def __str__(self):
        return f"<action_sequence: {[(round(a._longitude, 2), round(a._latitude, 2)) for a in self._action_sequence]}>"

    def __repr__(self):
        return self.__str__()

    @property
    def action_sequence(self):
        """Returns the sequence of primitive actions characterizing
        the macro_action.
        """
        return self._action_sequence

cdef class Nav3DObservation(Observation):

    NULL_OBS = ('', '')
    EPSILON=2.0
    cdef public tuple _pos_reading
    # cdef public int _hash
    def __init__(self, pos_reading):
        """
        pos_reading: The position reading of the observation.
        """
        self._pos_reading = pos_reading
        # self._hash = hash(self._pos_reading)

    def __hash__(self):
        # Keep this set to 1 if you want to identify observations via __eq__.
        # Python dictionaries identify keys if (__hash__ and __eq__) is True.
        return 1
        # return hash(self._pos_reading)

    def __eq__(self, other):
        if isinstance(other, Nav3DObservation):
            # return self._hash == other._hash
            if self._pos_reading == self.NULL_OBS or other._pos_reading == self.NULL_OBS:
                return self._pos_reading == other._pos_reading
            return np.linalg.norm(self.to_vector - other.to_vector, ord=2) < self.EPSILON
        return False

    def __str__(self):
        return f"<pos_reading: {self._pos_reading}>"

    def __repr__(self):
        return self.__str__()

    @property
    def to_vector(self):
        return np.array(self._pos_reading)

cdef class Nav3DMacroObservation(MacroObservation):

    CHECK_FREQ = 1
    LEN_TOLERANCE = 3
    cdef public list _observation_sequence

    def __init__(self, observation_sequence):

        self._observation_sequence = observation_sequence

    def __hash__(self):
        return 1

    def __eq__(self, other):
        if isinstance(other, Nav3DMacroObservation):
            if abs(len(self.observation_sequence) - len(other.observation_sequence)) <= self.LEN_TOLERANCE:
                shorter_length = min(len(self.observation_sequence), len(other.observation_sequence))
                return all([self.observation_sequence[i] == other.observation_sequence[i] for i in range(0, shorter_length, self.CHECK_FREQ)])

    def __str__(self):
        return f"<obs_sequence: {[o._pos_reading for o in self.observation_sequence]}>"

    def __repr__(self):
        return self.__str__()

    @property
    def observation_sequence(self):
        return self._observation_sequence

cdef class Nav3DTransitionModel(TransitionModel):

    STEP_SIZE = 1.0

    MEAN = list(np.array([0, 0, 0]))
    COV = list(np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]]) * 0.02 * STEP_SIZE)

    CONTROL_ERROR_DIST = pomdp_py.Gaussian(mean=MEAN, cov=COV)

    cdef public PyBulletEnv _pyb_env
    def __init__(self, pyb_env):
        """
        pyb_env (PyBulletEnv): A PyBullet environment instance as defined in examples.motion_planning.environments.
        """
        self._pyb_env = pyb_env

    cpdef sample(self, Nav3DState state, Nav3DAction action):

        if state.terminal:
            return state

        cdef tuple next_position

        dv = self.STEP_SIZE * np.array([np.cos(action._latitude) * np.cos(action._longitude),
                                       np.cos(action._latitude) * np.sin(action._longitude),
                                       np.sin(action._latitude)]) + self.CONTROL_ERROR_DIST.random()

        next_position = state._position

        if self._pyb_env.collision_checker(tuple(np.array(next_position) + np.array([1, 0, 0]) * dv) + (0, 0, 0)):
            dv[0] = 0
        if self._pyb_env.collision_checker(tuple(np.array(next_position) + np.array([0, 1, 0]) * dv) + (0, 0, 0)):
            dv[1] = 0
        if self._pyb_env.collision_checker(tuple(np.array(next_position) + np.array([0, 0, 1]) * dv) + (0, 0, 0)):
            dv[2] = 0

        next_position = tuple(np.array(next_position) + dv)

        return Nav3DState(next_position,
                     self._pyb_env.dz_checker(next_position + (0, 0, 0)),
                     self._pyb_env.lm_checker(next_position + (0, 0, 0)),
                     self._pyb_env.goal_checker(next_position + (0, 0, 0)))

cdef class Nav3DObservationModel(ObservationModel):

    #TODO: Remove randomness?

    MEAN = list(np.array([0, 0, 0]))
    COV = list(np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]]) * 0.01)

    cdef public PyBulletEnv _pyb_env
    def __init__(self, pyb_env):
        """
        pyb_env (PyBulletEnv): A PyBullet environment instance as defined in examples.motion_planning.environments.
        """
        self._pyb_env = pyb_env

    def probability(self, Nav3DObservation observation, Nav3DState next_state, Nav3DAction action):

        if self._pyb_env.lm_checker(next_state._position + (0, 0, 0)):
            if observation._pos_reading == Nav3DObservation.NULL_OBS:
                return 0
            else:
                if max(abs(np.array(next_state._position) - np.array(observation._pos_reading))) <= 1.5: # TODO: hardcoded.
                    return 1
                else:
                    return 0
        else:
            if observation._pos_reading == Nav3DObservation.NULL_OBS:
                return 1
            else:
                return 0

    def sample(self, Nav3DState next_state, Nav3DAction action):

        if not self._pyb_env.lm_checker(next_state._position + (0,0,0)):
            return Nav3DObservation(pos_reading=Nav3DObservation.NULL_OBS)

        return Nav3DObservation(pos_reading=tuple(map(lambda x: round(x, 0), next_state._position)))

cdef class Nav3DRewardModel(RewardModel):

    cdef public PyBulletEnv _pyb_env

    STEP_REWARD = -5  # The step penalty for non-terminal states.
    DZ_REWARD = -500  # The collision penalty.
    GOAL_REWARD = 2000  # The goal reward.

    def __init__(self, pyb_env):
        """
        pyb_env (PyBulletEnv): A PyBullet environment instance as defined in examples.motion_planning.environments.
        """
        self._pyb_env = pyb_env

    def probability(self, reward, state, action, next_state):

        raise NotImplementedError

    def sample(self, Nav3DState state, Nav3DAction action, Nav3DState next_state):

        if state.terminal:
            return 0.0

        if next_state._danger_zone:
            return self.DZ_REWARD

        if next_state._goal:
            return self.GOAL_REWARD

        return self.STEP_REWARD

cdef class Nav3DBlackboxModel(BlackboxModel):

    cdef public PyBulletEnv _pyb_env
    cdef public Nav3DTransitionModel _Tm
    cdef public Nav3DObservationModel _Om
    cdef public Nav3DRewardModel _Rm

    def __init__(self, PyBulletEnv pyb_env):

        self._pyb_env = pyb_env
        self._Tm = Nav3DTransitionModel(pyb_env)
        self._Om = Nav3DObservationModel(pyb_env)
        self._Rm = Nav3DRewardModel(pyb_env)

    def sample(self, State state, Action action, discount_factor=1.0):
        ns, oo, rr, nsteps = pomdp_py.sample_explict_models(T=self._Tm, O=self._Om, R=self._Rm,
                                                            state=state, action=action,
                                                            discount_factor=discount_factor)

        return ns, Nav3DMacroObservation(oo) if isinstance(oo, list) else oo, rr, nsteps

class Nav3DMacroReferencePolicyModel(PolicyModel):

    REACHED_DIST = 2.

    def __init__(self, pyb_env: PyBulletEnv, max_macro_action_length=1,
                 prm_nodes=500, prm_assert_connected=False):

        milestones = [
            (-26, -20, -5),
            (-20, -25, -5),
            (2, -9, -5),
            (7, 8, 0),
            (17, -7, 6),
            (27, 10, -4)
        ]

        goals = [
            (17, -7, 6),
            (27, 10, -4)
        ]

        critical_junctures = [
            (10, 8, 0),
            (29, -29, 0),
            (12, 8, 0),
            (-27, 0, 0),
            (-27, 8, 0),
            (-27, 0, -6),
            (-27, 8, -6),
            (-27, -10, 0),
            (-27, -10, -6),
            (-21, -10, -5),
            (-18, -10, -5)
        ]

        key_positions = milestones + goals + critical_junctures

        self._pyb_env = pyb_env
        self._Tm = Nav3DTransitionModel(pyb_env)
        self._Rm = Nav3DRewardModel(pyb_env)
        self._prm = PRM(pyb_env,
                            3,
                            [-30, -30, -10],
                            [30, 30, 10],
                            prm_nodes,
                            10,
                            max_degree=50,
                            sampling_dist="Gaussian",
                            gaussian_noise=10,
                            nodes_to_include=key_positions,
                            assert_connected=prm_assert_connected)
        self._max_macro_action_length = max_macro_action_length
        self._discrete_actions = self.enumerate_discrete_actions()

    def new_global_objective(self, state=None):
        target = self.get_new_target(state)
        targets = [self.get_new_waypoint(), target]
        kwargs = {"targets_reached": [False, False]}
        return Objective(states=targets, kwargs=kwargs)

    def target_reached(self, Nav3DState state, tuple target):
        return np.linalg.norm(np.array(state._position + (0,0,0)) - np.array(
                target)) < self.REACHED_DIST

    def maintain_global_objective(self, Nav3DState state, Objective objective):
        """If the objective has been attained, sample a new one.
        Otherwise, maintain the objective by keeping track of the statuses."""

        if objective is None or all(objective.kwargs.get("targets_reached")):
            return self.new_global_objective()

        if self.target_reached(state, objective.states[0]) and objective.kwargs.get("targets_reached") == [False, False]:
            return Objective(states=objective.states, kwargs={"targets_reached":[True, False]})
        elif self.target_reached(state, objective.states[1]) and objective.kwargs.get("targets_reached") == [True, False]:
            return Objective(states=objective.states, kwargs={"targets_reached":[True, True]})
        else:
            return objective

    def sample(self, Nav3DState state, Objective objective):


        target = objective.states[1] if objective.kwargs.get("targets_reached")[0] else objective.states[0]
        path = self._prm.shortest_path_offline(state._position+(0,0,0), target)

        if len(path) < 2:
            # print("No path to target node in PRM. "
            #       + "Taking the straight line distance irrespective of obstacles.")
            # NB: this just takes the straight line irrespective of obstacles.
            edges = [(state._position + (0,0,0), target)]
        else:
            edges = [(path[i], path[i+1]) for i in range(len(path) - 1)]
            # edges = [(path[0], path[2])]

        # action_sequence = []
        #
        # e = edges[0]
        # edge_vector = np.array(e[1]) - np.array(e[0])
        # edge_distance = np.linalg.norm(edge_vector)
        # action = Nav3DAction(np.arctan2(edge_vector[1], edge_vector[0]), np.arcsin(edge_vector[2]/edge_distance))
        # m = int(np.ceil(edge_distance / self._Tm.STEP_SIZE))
        # for j in range(m):
        #     action_sequence.append(action)
        #     if len(action_sequence) >= self._max_macro_action_length:
        #         return Nav3DMacroAction(action_sequence)
        #
        # return Nav3DMacroAction(action_sequence)

        # e = self._prm.sample_out_edge(state._position + (0, 0, 0))
        # action_sequence = []
        # edge_vector = np.array(e[1]) - np.array(e[0])
        # edge_distance = np.linalg.norm(edge_vector)
        # if edge_distance== 0:
        #     edge_distance = 1
        #     print(e)
        #     print("Adjusted edge_distance to 1.")
        # action = Nav3DAction(np.arctan2(edge_vector[1], edge_vector[0]), np.arcsin(edge_vector[2]/edge_distance))
        # m = int(np.ceil(edge_distance / self._Tm.STEP_SIZE))
        # for j in range(m):
        #     action_sequence.append(action)
        #     if len(action_sequence) >= self._max_macro_action_length:
        #         return Nav3DMacroAction(action_sequence)

        # target = objective.states[1] if objective.kwargs.get("targets_reached")[0] else objective.states[0]
        # path = self._prm.shortest_path_offline(np.array(state._position+(0,0,0)), target)
        #
        # if len(path) < 2:
        #     if len(path) < 2:
        #         edges = [(state._position + (0, 0, 0), target)]
        #     else:
        #         edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        #
        action_sequence = []

        i = 0
        for e in edges:
            edge_vector = np.array(e[1]) - np.array(e[0])
            edge_distance = np.linalg.norm(edge_vector)
            action = Nav3DAction(np.arctan2(edge_vector[1], edge_vector[0]), np.arcsin(edge_vector[2]/edge_distance))
            m = int(np.ceil(edge_distance / self._Tm.STEP_SIZE))
            for j in range(m):
                action_sequence.append(action)
                if len(action_sequence) >= self._max_macro_action_length:
                    return Nav3DMacroAction(action_sequence)

        return Nav3DMacroAction(action_sequence)



    def value_heuristic(self, Nav3DState state, Nav3DState next_state, float reward, float discount):

        if next_state is not None:
            if next_state.terminal:
                # print("V Heur (terminal) | ", reward)
                return reward
        d = int(self.distance_to_goal(state)/self._Tm.STEP_SIZE)
        # print("V Heur | ", (self._Rm.STEP_REWARD * (1 - discount**d)/(1 - discount)) +  + self._Rm.GOAL_REWARD*(discount**d))
        return (self._Rm.STEP_REWARD * (1 - discount**d)/(1 - discount)) + self._Rm.GOAL_REWARD*(discount**d) / 5

    def get_new_target(self, Nav3DState state):
        return self._pyb_env.sample_goal_config()

    def get_new_waypoint(self, prob_key_config=1.0, lower_bounds=(-30, -30, -9), upper_bounds=(30, 30, 9)):
        "With probability prob_key_config, sample a key_config; otherwise uniformly sample a collision-free point."
        if np.random.random() < prob_key_config:
            return self._pyb_env.sample_key_config()
        x, y, z = np.random.uniform(low=lower_bounds, high=upper_bounds, size=(1, 3))[0]
        nearest_d, nearest_i = self._prm.kd_tree.query((x, y, z, 0, 0, 0))
        return tuple(self._prm.kd_tree.data[nearest_i])

    def rollout(self, Nav3DState state, history=None):

        e = self._prm.sample_out_edge(state._position + (0, 0, 0))
        action_sequence = []
        edge_vector = np.array(e[1]) - np.array(e[0])
        edge_distance = np.linalg.norm(edge_vector)
        action = Nav3DAction(np.arctan2(edge_vector[1], edge_vector[0]), np.arcsin(edge_vector[2]/edge_distance))
        m = int(np.ceil(edge_distance / self._Tm.STEP_SIZE))
        for j in range(m):
            action_sequence.append(action)
            if len(action_sequence) >= self._max_macro_action_length:
                return Nav3DMacroAction(action_sequence)

        return Nav3DMacroAction(action_sequence)

        # path = self._prm.shortest_path_offline(np.array(state._position + (0, 0, 0)),
        #                                        self._pyb_env.sample_goal_configs())
        # edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        # action_sequence = []
        #
        # i = 0
        # for e in edges:
        #     edge_vector = np.array(e[1]) - np.array(e[0])
        #     edge_distance = np.linalg.norm(edge_vector)
        #     action = Nav3DAction(np.arctan2(edge_vector[1], edge_vector[0]), np.arcsin(edge_vector[2]/edge_distance))
        #     m = int(np.ceil(edge_distance / self._Tm.STEP_SIZE))
        #     for j in range(m):
        #         action_sequence.append(action)
        #         if len(action_sequence) >= self._max_macro_action_length:
        #             return Nav3DMacroAction(action_sequence)
        #
        # return Nav3DMacroAction(action_sequence)

    def distance_to_goal(self, Nav3DState state):

        dist = np.inf
        for goal in self._pyb_env.goal_configs:
            dist_to_goal = self._prm.shortest_path_length(np.array(state._position + (0, 0, 0)),
                                                          np.array(goal))
            if dist_to_goal < dist:
                dist = dist_to_goal

        return dist

    def enumerate_discrete_actions(self, n=4, m=3):
        longi_list = [-np.pi + i * np.pi * 2 / n for i in range(0, n)]
        lati_list = [-np.pi/2 + j * np.pi / m for j in range(0, m+1)]
        longi_list.sort(key=abs)
        lati_list.sort(key=abs)

        actions = []
        for i in longi_list:
            for j in lati_list:
                actions.append(Nav3DMacroAction([Nav3DAction(i, j)]*self._max_macro_action_length))

        return actions

    def get_all_actions(self, state=None, history=None):
        return self._discrete_actions

    def rollout(self, state, history=None):
        return np.random.choice(self.get_all_actions(state=state, history=history))


# OLD REFERENCE POLICY
# class Nav3DMacroReferencePolicyModel(PolicyModel):
#
#     REACHED_DIST = 2.
#
#     def __init__(self, pyb_env: PyBulletEnv, max_macro_action_length=1,
#                  prm_nodes=500, prm_assert_connected=False):
#
#         milestones = [
#             (-26, -20, -5),
#             (-20, -25, -5),
#             (2, -9, -5),
#             (7, 8, 0),
#             (17, -7, 6),
#             (27, 10, -4)
#         ]
#
#         goals = [
#             (17, -7, 6),
#             (27, 10, -4)
#         ]
#
#         critical_junctures = [
#             (10, 8, 0),
#             (29, -29, 0),
#             (12, 8, 0),
#             (-27, 0, 0),
#             (-27, 8, 0),
#             (-27, 0, -6),
#             (-27, 8, -6),
#             (-27, -10, 0),
#             (-27, -10, -6),
#             (-21, -10, -5),
#             (-18, -10, -5)
#         ]
#
#         key_positions = milestones + goals + critical_junctures
#
#         self._pyb_env = pyb_env
#         self._Tm = Nav3DTransitionModel(pyb_env)
#         self._Rm = Nav3DRewardModel(pyb_env)
#         self._prm = PRM(pyb_env,
#                             3,
#                             [-30, -30, -10],
#                             [30, 30, 10],
#                             prm_nodes,
#                             20,
#                             max_degree=50,
#                             sampling_dist="Uniform",
#                             gaussian_noise=10,
#                             nodes_to_include=key_positions,
#                             assert_connected=prm_assert_connected)
#         self._max_macro_action_length = max_macro_action_length
#         self._discrete_actions = self.enumerate_discrete_actions()
#
#     def new_global_objective(self, state=None):
#         target = self.get_new_target(state)
#         targets = [self.get_new_waypoint(), target]
#         kwargs = {"targets_reached": [False, False]}
#         return Objective(states=targets, kwargs=kwargs)
#
#     def target_reached(self, Nav3DState state, tuple target):
#         return np.linalg.norm(np.array(state._position + (0,0,0)) - np.array(
#                 target)) < self.REACHED_DIST
#
#     def maintain_global_objective(self, Nav3DState state, Objective objective):
#         """If the objective has been attained, sample a new one.
#         Otherwise, maintain the objective by keeping track of the statuses."""
#
#         if objective is None or all(objective.kwargs.get("targets_reached")):
#             return self.new_global_objective()
#
#         if self.target_reached(state, objective.states[0]) and objective.kwargs.get("targets_reached") == [False, False]:
#             return Objective(states=objective.states, kwargs={"targets_reached":[True, False]})
#         elif self.target_reached(state, objective.states[1]) and objective.kwargs.get("targets_reached") == [True, False]:
#             return Objective(states=objective.states, kwargs={"targets_reached":[True, True]})
#         else:
#             return objective
#
#     def sample(self, Nav3DState state, Objective objective):
#
#         target = objective.states[1] if objective.kwargs.get("targets_reached")[0] else objective.states[0]
#         path = self._prm.shortest_path_offline(state._position+(0,0,0), target)
#
#         if len(path) < 2:
#             # print("No path to target node in PRM. "
#             #       + "Taking the straight line distance irrespective of obstacles.")
#             # NB: this just takes the straight line irrespective of obstacles.
#             edges = [(state._position + (0,0,0), target)]
#         else:
#             edges = [(path[i], path[i+1]) for i in range(len(path) - 1)]
#             # edges = [(path[0], path[1])]
#
#         action_sequence = []
#
#         i = 0
#         for e in edges:
#             edge_vector = np.array(e[1]) - np.array(e[0])
#             edge_distance = np.linalg.norm(edge_vector)
#             action = Nav3DAction(np.arctan2(edge_vector[1], edge_vector[0]), np.arcsin(edge_vector[2]/edge_distance))
#             m = int(np.ceil(edge_distance / self._Tm.STEP_SIZE))
#             for j in range(m):
#                 action_sequence.append(action)
#                 if len(action_sequence) >= self._max_macro_action_length:
#                     return Nav3DMacroAction(action_sequence)
#
#         # e = edges[0]
#         # edge_vector = np.array(e[1]) - np.array(e[0])
#         # edge_distance = np.linalg.norm(edge_vector)
#         # action = Nav3DAction(np.arctan2(edge_vector[1], edge_vector[0]), np.arcsin(edge_vector[2]/edge_distance))
#         # m = int(np.ceil(edge_distance / self._Tm.STEP_SIZE))
#         # for j in range(m):
#         #     action_sequence.append(action)
#         #     if len(action_sequence) >= self._max_macro_action_length:
#         #         return Nav3DMacroAction(action_sequence)
#
#         return Nav3DMacroAction(action_sequence)
#
#     def value_heuristic(self, Nav3DState state, Nav3DState next_state, float reward, float discount):
#
#         if next_state is not None:
#             if next_state.terminal:
#                 return reward
#         d = int(self.distance_to_goal(state)/self._Tm.STEP_SIZE)
#         return (self._Rm.STEP_REWARD * (1 - discount**d)/(1 - discount))
#
#     def get_new_target(self, Nav3DState state):
#         return self._pyb_env.sample_goal_config()
#
#     def get_new_waypoint(self, prob_key_config=1.0, lower_bounds=(-30, -30, -9), upper_bounds=(30, 30, 9)):
#         "With probability prob_key_config, sample a key_config; otherwise uniformly sample a collision-free point."
#         if np.random.random() < prob_key_config:
#             return self._pyb_env.sample_key_config()
#         x, y, z = np.random.uniform(low=lower_bounds, high=upper_bounds, size=(1, 3))[0]
#         nearest_d, nearest_i = self._prm.kd_tree.query((x, y, z, 0, 0, 0))
#         return tuple(self._prm.kd_tree.data[nearest_i])
#
#     def rollout(self, Nav3DState state, history=None):
#
#         path = self._prm.shortest_path_offline(np.array(state._position + (0, 0, 0)),
#                                                self._pyb_env.sample_goal_configs())
#         edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
#         action_sequence = []
#
#         i = 0
#         for e in edges:
#             edge_vector = np.array(e[1]) - np.array(e[0])
#             edge_distance = np.linalg.norm(edge_vector)
#             action = Nav3DAction(np.arctan2(edge_vector[1], edge_vector[0]), np.arcsin(edge_vector[2]/edge_distance))
#             m = int(np.ceil(edge_distance / self._Tm.STEP_SIZE))
#             for j in range(m):
#                 action_sequence.append(action)
#                 if len(action_sequence) >= self._max_macro_action_length:
#                     return Nav3DMacroAction(action_sequence)
#
#         return Nav3DMacroAction(action_sequence)
#
#     def distance_to_goal(self, Nav3DState state):
#
#         dist = np.inf
#         for goal in self._pyb_env.goal_configs:
#             dist_to_goal = self._prm.shortest_path_length(np.array(state._position + (0, 0, 0)),
#                                                           np.array(goal))
#             if dist_to_goal < dist:
#                 dist = dist_to_goal
#
#         return dist
#
#     def enumerate_discrete_actions(self, n=4, m=3):
#         longi_list = [-np.pi + i * np.pi * 2 / n for i in range(0, n)]
#         lati_list = [-np.pi/2 + j * np.pi / m for j in range(0, m+1)]
#         longi_list.sort(key=abs)
#         lati_list.sort(key=abs)
#
#         actions = []
#         for i in longi_list:
#             for j in lati_list:
#                 actions.append(Nav3DMacroAction([Nav3DAction(i, j)]*self._max_macro_action_length))
#
#         return actions
#
#     def get_all_actions(self, state=None, history=None):
#         return self._discrete_actions
#
#     def rollout(self, state, history=None):
#         return np.random.choice(self.get_all_actions(state=state, history=history))

class Nav3DContinuousPOMDP(MPPOMDP):
    def __init__(self,
                 init_state,
                 init_belief,
                 pyb_env,
                 pyb_env_gui,
                 max_macro_action_length=1,
                 prm_nodes=500,
                 prm_assert_connected=False):
        self._init_state = init_state
        self._init_belief = init_belief,
        self._pyb_env = pyb_env
        self._pyb_env_gui = pyb_env_gui

        "Agent"
        agent = pomdp_py.Agent(init_belief=init_belief,
                             policy_model=Nav3DMacroReferencePolicyModel(pyb_env=pyb_env,
                                                                      max_macro_action_length=max_macro_action_length,
                                                                      prm_nodes=prm_nodes, prm_assert_connected=prm_assert_connected),
                             transition_model=Nav3DTransitionModel(pyb_env=pyb_env),
                             observation_model=Nav3DObservationModel(pyb_env=pyb_env),
                             reward_model=Nav3DRewardModel(pyb_env=pyb_env),
                             blackbox_model=Nav3DBlackboxModel(pyb_env=pyb_env),
                             name=f"Nav3DContinuousAgent({max_macro_action_length})")

        "Environment"
        env = pomdp_py.Environment(
            init_state=init_state,
            transition_model=Nav3DTransitionModel(pyb_env=pyb_env_gui),
            reward_model=Nav3DRewardModel(pyb_env=pyb_env_gui))

        super().__init__(agent, env, name="Nav3DContinuousPOMDP")

    def visualize_belief(self, histogram, timeout=10., life_time=.1, rgb=(0,1,0)):
        robots = []
        elapsed_time = 0.
        for state, prob in histogram.items():
            start_time = time.time()

            robot_id = pyb.loadURDF(
                str((pathlib.Path(__file__).parent.parent.parent.parent / "data/cuboid.urdf")),
                state._position,
                pyb.getQuaternionFromEuler((0, 0, 0)),
                globalScaling=.8,
                physicsClientId=self._pyb_env_gui._id
            )
            pyb.changeVisualShape(robot_id, -1, rgbaColor=rgb+(prob*0.7 + 0.3,),
                                  physicsClientId=self._pyb_env_gui._id)
            robots.append(robot_id)

            end_time = time.time()
            elapsed_time += (end_time - start_time)
            if elapsed_time >= timeout:
                break

        time.sleep(life_time)
        for robot_id in robots:
            pyb.changeVisualShape(robot_id, -1, rgbaColor=[0, 0, 0, 0], physicsClientId=self._pyb_env_gui._id)

    def visualize_world(self):
        self._pyb_env_gui.set_config(self.env.state._position + (0,0,0))
