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

cdef class Nav2DState(State):
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
        if isinstance(other, Nav2DState):
            return (np.linalg.norm(np.array(self._position) - np.array(other._position), ord=2) < Nav2DState.EPSILON and
                    self._terminal == other._terminal and
                    self._danger_zone == other._danger_zone and
                    self._landmark == other._landmark and
                    self._goal == other._goal)
        else:
            return False

    def __str__(self):
        return f"<pos: ({self._position[0]:.3f}, {self._position[1]:.3f}) | dz: {self._danger_zone} | lm: {self._landmark} | goal: {self._goal}>"

    def __repr__(self):
        return self.__str__()

    @property
    def xyz(self):
        return np.array(self._position+(0,))

    @property
    def terminal(self):
        return self._terminal

    @property
    def is_goal(self):
        return self._goal

cdef class Nav2DAction(Action):

    EPSILON=np.pi/12
    cdef public float _angle

    def __init__(self, angle):
        """
        angle (float): The directional angle.
        """

        if angle > np.pi or angle < -np.pi:
            raise(ValueError, "Make sure angle is in radians and in the principal range.")

        self._angle = angle
        # self._hash = hash(angle)

    def __hash__(self):
        # Keep this set to 1 if you want to identify actions via __eq__.
        # Python dictionaries identify keys if (__hash__ and __eq__) is True.
        return 1
        # return self._hash

    def __eq__(self, other):
        if isinstance(other, Action):
            return min(2*np.pi - abs(self._angle - other._angle), abs(self._angle - other._angle)) <= self.EPSILON
        return False

    def __str__(self):
        return f"<angle (rad): {self._angle:.3f}>"

    def __repr__(self):
        return self.__str__()

cdef class Nav2DMacroAction(MacroAction):

    EPSILON_ANGLE = np.pi/6
    EPSILON_EFFECT = 3

    def __init__(self, action_sequence):
        """
        action_sequence (list): A list of primitive actions
        """

        self._action_sequence = action_sequence
        # self._hash = hash(tuple(action_sequence))

    def to_vector(self, Nav2DAction action):
        return np.array([np.cos(action._angle), np.sin(action._angle)])

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

    def __eq__(self, Nav2DMacroAction other):
        if isinstance(other, Nav2DMacroAction):
            if self.angle_between(self.effect(), other.effect()) < self.EPSILON_ANGLE and np.linalg.norm(self.effect() - other.effect()) <= self.EPSILON_EFFECT:
                return True
            return False
        return False

    def __str__(self):
        return f"<action_sequence: {[round(a._angle, 3) for a in self._action_sequence]}>"

    def __repr__(self):
        return self.__str__()

    @property
    def action_sequence(self):
        """Returns the sequence of primitive actions characterizing
        the macro_action.
        """
        return self._action_sequence

cdef class Nav2DObservation(Observation):

    NULL_OBS = ('', '')
    EPSILON = 2.0
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
        if isinstance(other, Nav2DObservation):
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

cdef class Nav2DMacroObservation(MacroObservation):

    CHECK_FREQ = 1
    LEN_TOLERANCE = 1
    cdef public list _observation_sequence

    def __init__(self, observation_sequence):

        self._observation_sequence = observation_sequence

    def __hash__(self):
        return 1

    def __eq__(self, other):
        if isinstance(other, Nav2DMacroObservation):
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

cdef class Nav2DTransitionModel(TransitionModel):

    STEP_SIZE = 1.0

    MEAN = list(np.array([0, 0]))
    COV = list(np.array([[1, 0],
                    [0, 1]]) * 0.05 * STEP_SIZE)

    CONTROL_ERROR_DIST = pomdp_py.Gaussian(mean=MEAN, cov=COV)

    cdef public PyBulletEnv _pyb_env
    def __init__(self, pyb_env):
        """
        pyb_env (PyBulletEnv): A PyBullet environment instance as defined in examples.motion_planning.environments.
        """
        self._pyb_env = pyb_env

    cpdef move_if_possible(self, Nav2DState state, position_delta):
        cdef tuple potential_next_position

        potential_next_position = tuple(np.array(state._position) + position_delta + self.CONTROL_ERROR_DIST.random())

        if self._pyb_env.collision_checker(potential_next_position + (0, 0, 0, 0)):
            return state

        return Nav2DState(potential_next_position,
                     self._pyb_env.dz_checker(potential_next_position + (0, 0, 0, 0)),
                     self._pyb_env.lm_checker(potential_next_position + (0, 0, 0, 0)),
                     self._pyb_env.goal_checker(potential_next_position + (0, 0, 0, 0)))


    cpdef sample(self, Nav2DState state, Nav2DAction action):

        # if state.terminal:
        #     return state
        #
        # d = self.STEP_SIZE * np.array((np.cos(action._angle), np.sin(action._angle)))
        #
        # return self.move_if_possible(state, d)

        if state.terminal:
            return state

        cdef float epsilon = np.pi / 24 * np.random.choice([-1, 1])
        cdef float angle_delta = 0
        cdef tuple potential_next_position

        d = self.STEP_SIZE * np.array((np.cos(action._angle), np.sin(action._angle)))
        potential_next_position = tuple(np.array(state._position) + d + self.CONTROL_ERROR_DIST.random())

        while self._pyb_env.collision_checker(potential_next_position + (0, 0, 0, 0)):
            angle_delta += epsilon
            d = self.STEP_SIZE * np.array((np.cos(action._angle+angle_delta), np.sin(action._angle+angle_delta)))
            potential_next_position = tuple(np.array(state._position) + d + self.CONTROL_ERROR_DIST.random())

        return Nav2DState(potential_next_position,
                          self._pyb_env.dz_checker(potential_next_position + (0, 0, 0, 0)),
                          self._pyb_env.lm_checker(potential_next_position + (0, 0, 0, 0)),
                          self._pyb_env.goal_checker(potential_next_position + (0, 0, 0, 0)))

cdef class Nav2DObservationModel(ObservationModel):

    MEAN = list(np.array([0, 0]))
    COV = list(np.array([[1, 0],
                    [0, 1]]) * 0.01)

    cdef public PyBulletEnv _pyb_env
    def __init__(self, pyb_env):
        """
        pyb_env (PyBulletEnv): A PyBullet environment instance as defined in examples.motion_planning.environments.
        """
        self._pyb_env = pyb_env

    def probability(self, Nav2DObservation observation, Nav2DState next_state, Nav2DAction action):
        if observation._pos_reading == Nav2DObservation.NULL_OBS:
            if self._pyb_env.lm_checker(next_state._position + (0, 0, 0, 0)):
                return -np.inf # because _sir in update uses the softmax.
            else:
                return 1.
        else:
            error_dist = pomdp_py.Gaussian(mean=list(next_state._position), cov=self.COV)
            return error_dist[observation.to_vector]

    def sample(self, Nav2DState next_state, Nav2DAction action):

        if not self._pyb_env.lm_checker(next_state._position + (0,0,0,0)):
            return Nav2DObservation(pos_reading=Nav2DObservation.NULL_OBS)

        return Nav2DObservation(pos_reading=tuple(map(lambda x: round(x, 0), next_state._position)))

cdef class Nav2DRewardModel(RewardModel):

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

    def sample(self, Nav2DState state, Nav2DAction action, Nav2DState next_state):

        if state.terminal:
            return 0.0

        if next_state._danger_zone:
            return self.DZ_REWARD

        if next_state._goal:
            return self.GOAL_REWARD

        return self.STEP_REWARD

cdef class Nav2DBlackboxModel(BlackboxModel):

    cdef public PyBulletEnv _pyb_env
    cdef public Nav2DTransitionModel _Tm
    cdef public Nav2DObservationModel _Om
    cdef public Nav2DRewardModel _Rm

    def __init__(self, PyBulletEnv pyb_env):

        self._pyb_env = pyb_env
        self._Tm = Nav2DTransitionModel(pyb_env)
        self._Om = Nav2DObservationModel(pyb_env)
        self._Rm = Nav2DRewardModel(pyb_env)

    def sample(self, State state, Action action, discount_factor=1.0):
        ns, oo, rr, nsteps = pomdp_py.sample_explict_models(T=self._Tm, O=self._Om, R=self._Rm,
                                                            state=state, action=action,
                                                            discount_factor=discount_factor)

        return ns, Nav2DMacroObservation(oo) if isinstance(oo, list) else oo, rr, nsteps

class Nav2DMacroReferencePolicyModel(PolicyModel):

    REACHED_DIST = 2.
    COMPASS_DIRECTIONS = list(np.array([0., .25, .5, .75, -.25, -.5, -.75, 1.]) * np.pi)

    def __init__(self, pyb_env: PyBulletEnv, max_macro_action_length=1):

        spawn_positions = [(1, -13), (1, -40)]
        milestones = ([(4, -52), (4, -60), (10, -58), (31, -40), (34, -46), (31, -58),
                     (40, -60), (37, -25), (43, -31), (55, -31)])
        goal_positions = [(58, -22), (49, -37)]
        others = [(20, -14), (16, -40), (2, -2), (37, -14), (3,-26),
                  (13, -10), (33, -10), (43, -22), (13, -55), (43, -22),
                  (2, -20), (2, -30), (27, -14), (14, -59), (7, -14), (34, -14),
                  (25, -42), (22, -58), (13, -41), (49, -30), (13, -14), (13, -52)]
        key_positions = spawn_positions + milestones + others

        self._pyb_env = pyb_env
        self._Tm = Nav2DTransitionModel(pyb_env)
        self._Rm = Nav2DRewardModel(pyb_env)
        self._prm = PRM(pyb_env,
                            2,
                            [0, -60],
                            [60, 0],
                            100,
                            15,
                            sampling_dist="Gaussian",
                            gaussian_noise=10,
                            nodes_to_include=key_positions)
        self._max_macro_action_length = max_macro_action_length

    # def new_global_objective(self, state=None):
    #     targets = [self._pyb_env.sample_key_config()]
    #     kwargs = {"target_reached": False}
    #     return Objective(states=targets, kwargs=kwargs)
    #
    # def maintain_global_objective(self, Nav2DState state, Objective objective):
    #     if objective is None or self.target_reached(state, objective.states[0]):
    #         return self.new_global_objective()
    #
    #     return objective

    def new_global_objective(self, state=None, objective=None):
        targets = [self._pyb_env.sample_key_config()] + [self._pyb_env.sample_goal_config()]
        kwargs = {"targets_reached" : [False, False]}
        return Objective(states=targets, kwargs=kwargs)

    def target_reached(self, Nav2DState state, target):
        return np.linalg.norm(np.array(state._position+(0,0,0,0)) - np.array(target)) <= self.REACHED_DIST

    def maintain_global_objective(self, Nav2DState state, Objective objective):
        if objective is None or all(objective.kwargs.get("targets_reached")):
            return self.new_global_objective()

        if self.target_reached(state, objective.states[0]) and objective.kwargs.get("targets_reached") == [False, False]:
            return Objective(states=objective.states, kwargs={"targets_reached":[True, False]})
        elif self.target_reached(state, objective.states[1]) and objective.kwargs.get("targets_reached") == [True, False]:
            return Objective(states=objective.states, kwargs={"targets_reached":[True, True]})
        else:
            return objective

    def value_heuristic(self, Nav2DState state, Nav2DState next_state, float reward, float discount):

        if next_state is not None:
            if next_state.terminal:
                # print("V Heur (terminal) | ", reward)
                return reward
        d = np.ceil(self.distance_to_goal(state)/self._Tm.STEP_SIZE)
        # print("V Heur | ", (self._Rm.STEP_REWARD * (1 - discount**d)/(1 - discount)) +  + self._Rm.GOAL_REWARD*(discount**d))
        return (self._Rm.STEP_REWARD * (1 - discount**d)/(1 - discount)) + self._Rm.GOAL_REWARD*(discount**d) / 10

    def sample(self, Nav2DState state, Objective objective):

        # e = self._prm.sample_out_edge(state._position + (0, 0, 0, 0))
        # action_sequence = []
        # edge_vector = np.array(e[1]) - np.array(e[0])
        # edge_distance = np.linalg.norm(edge_vector)
        # if edge_distance == 0:
        #     edge_distance = 1
        #     print(e)
        #     print("Adjusted edge_distance to 1.")
        # action = Nav2DAction(np.arctan2(edge_vector[1], edge_vector[0]))
        # m = int(np.ceil(edge_distance / self._Tm.STEP_SIZE))
        # for j in range(m):
        #     action_sequence.append(action)
        #     if len(action_sequence) >= self._max_macro_action_length:
        #         return Nav2DMacroAction(action_sequence)

        target = objective.states[1] if objective.kwargs.get("targets_reached")[0] else objective.states[0]
        # target = objective.states[0]
        path = self._prm.shortest_path_offline(np.array(state._position+(0,0,0,0)), target)

        if len(path) < 2:
            edges = [(state._position+(0,0,0,0), target)]
        else:
            edges = [(path[i], path[i+1]) for i in range(len(path) - 1)]
            # edges = [(path[0], path[1])]

        action_sequence = []

        i = 0
        for e in edges:
            edge_vector = np.array(e[1]) - np.array(e[0])
            action = Nav2DAction(np.arctan2(edge_vector[1], edge_vector[0]))
            edge_distance = np.linalg.norm(edge_vector)
            m = int(np.ceil(edge_distance / self._Tm.STEP_SIZE))
            for j in range(m):
                action_sequence.append(action)
                if len(action_sequence) >= self._max_macro_action_length:
                    return Nav2DMacroAction(action_sequence)

        # e = edges[0]
        # edge_vector = np.array(e[1]) - np.array(e[0])
        # action = Nav2DAction(np.arctan2(edge_vector[1], edge_vector[0]))
        # edge_distance = np.linalg.norm(edge_vector)
        # m = int(np.ceil(edge_distance / self._Tm.STEP_SIZE))
        # for j in range(m):
        #     action_sequence.append(action)
        #     if len(action_sequence) >= self._max_macro_action_length:
        #         return Nav2DMacroAction(action_sequence)

        return Nav2DMacroAction(action_sequence)

    def distance_to_goal(self, Nav2DState state):

        dist = np.inf

        for goal in self._pyb_env.goal_positions:
            dist_to_goal = self._prm.shortest_path_length(np.array(state._position + (0, 0, 0, 0)),
                                                          np.array(goal + (0, 0, 0, 0))) / self._Tm.STEP_SIZE
            if dist_to_goal < dist:
                dist = dist_to_goal

        return dist

    def get_all_actions(self, state=None, history=None):
        """Returns a list of macro actions composed of a uniform sequence of
        compass directions of required length."""
        return list(map(lambda a : Nav2DMacroAction([Nav2DAction(a)]*self._max_macro_action_length),
                        self.COMPASS_DIRECTIONS))

    def rollout(self, Nav2DState state, history=None):

        path = self._prm.shortest_path_offline(np.array(state._position + (0, 0, 0, 0)),
                                               self._pyb_env.sample_goal_config())
        edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        action_sequence = []

        i = 0
        for e in edges:
            edge_vector = np.array(e[1]) - np.array(e[0])
            action = Nav2DAction(np.arctan2(edge_vector[1], edge_vector[0]))
            edge_distance = np.linalg.norm(edge_vector)
            m = int(np.ceil(edge_distance / self._Tm.STEP_SIZE))
            for j in range(m):
                action_sequence.append(action)
                if len(action_sequence) >= self._max_macro_action_length:
                    return Nav2DMacroAction(action_sequence)

        return Nav2DMacroAction(action_sequence)

class Nav2DContinuousPOMDP(MPPOMDP):
    def __init__(self,
                 init_state,
                 init_belief,
                 pyb_env,
                 pyb_env_gui,
                 max_macro_action_length=1):
        self._pyb_env = pyb_env
        self._pyb_env_gui = pyb_env_gui

        "Agent"
        agent = pomdp_py.Agent(init_belief=init_belief,
                             policy_model=Nav2DMacroReferencePolicyModel(pyb_env=pyb_env,
                                                                      max_macro_action_length=max_macro_action_length),
                             transition_model=Nav2DTransitionModel(pyb_env=pyb_env),
                             observation_model=Nav2DObservationModel(pyb_env=pyb_env),
                             reward_model=Nav2DRewardModel(pyb_env=pyb_env),
                             blackbox_model=Nav2DBlackboxModel(pyb_env=pyb_env),
                             name=f"Nav2DContinuousAgent({max_macro_action_length})")

        "Environment"
        env = pomdp_py.Environment(
            init_state=init_state,
            transition_model=Nav2DTransitionModel(pyb_env=pyb_env_gui),
            reward_model=Nav2DRewardModel(pyb_env=pyb_env_gui))

        super().__init__(agent, env, name="Nav2DContinuousPOMDP")

    def visualize_belief(self, histogram, timeout=10., life_time=.1, rgb=(1,0,0)):
        robots = []
        elapsed_time = 0.
        for state, prob in histogram.items():
            start_time = time.time()

            robot_id = pyb.loadURDF(
                str((pathlib.Path(__file__).parent.parent.parent.parent / "data/cuboid.urdf")),
                state._position + (0,),
                pyb.getQuaternionFromEuler((0, 0, 0)),
                globalScaling=.8,
                physicsClientId=self._pyb_env_gui._id
            )
            pyb.changeVisualShape(robot_id, -1, rgbaColor=rgb+(prob*0.95 + 0.05,),
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
        self._pyb_env_gui.set_config(self.env.state._position + (0,0,0,0))