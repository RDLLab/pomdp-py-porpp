"""This file provides all plugins for the POMDP model.
"""

from pomdp_py.framework.basics cimport *
from pomdp_py.algorithms.po_uct cimport *
from pomdp_py.representations.belief.particles cimport *
from pomdp_py.problems.motion_planning.environments cimport *
from pomdp_py.problems.motion_planning.environments.pyb_env cimport PyBulletEnv

import pomdp_py
from pomdp_py.problems.motion_planning.path_planning.prm import PRM
from pomdp_py.problems.motion_planning.environments.corsica import Corsica
import numpy as np
import pybullet as pyb
import time
import pathlib
import copy

np.set_printoptions(precision=3, suppress=True)  # for neat printing of numpy arrays.

cdef angle_distance(float angle1, float angle2):

    return min(2*np.pi - abs(angle1 - angle2), abs(angle1 - angle2))

cdef class Nav3DState(State):

    EPSILON = 1. # For histogram.
    cdef public tuple _position
    cdef public bint _collision
    cdef public bint _no_fly_zone
    cdef public bint _landmark
    cdef public list _objectives_reached
    cdef public bint _goal
    cdef public bint _terminal

    def __init__(self, position, collision, no_fly_zone, landmark, objectives_reached, goal):
        """
        position (numpy array): The state position.
        collision (bool): The robot has collided with an obstacle.
        no_fly_zone (bool): The robot is in a no fly zone.
        landmark (bool): The robot is at a landmark.
        objectives_reached (list): A list of the sequenced objectives that robot has completed.
        goal (bool): The robot has completed its objective.
        """
        self._position = position
        self._collision = collision
        self._no_fly_zone = no_fly_zone
        self._landmark = landmark
        self._objectives_reached = objectives_reached
        self._goal = goal
        self._terminal = goal or collision

    def __hash__(self):
        # Keep this set to 1 if you want to identify states via __eq__.
        # Python dictionaries identify keys if (__hash__ and __eq__) is True.
        return 1

    def __eq__(self, other):
        if not isinstance(other, Nav3DState):
            return False
        return (np.linalg.norm(np.array(self._position) - np.array(other._position), ord=2) < Nav3DState.EPSILON and
                self._collision == other._collision and
                self._no_fly_zone == other._no_fly_zone and
                self._landmark == other._landmark and
                set(self._objectives_reached) == set(other._objectives_reached) and
                self._terminal == other._terminal)

    def __str__(self):
        return (f"<pos: ({self._position[0]:.3f}, {self._position[1]:.3f}, {self._position[2]:.3f}) "
                f"| col: {self._collision} | nfz: {self._no_fly_zone} | lm: {self._landmark} "
                f"| objectives_reached: {self._objectives_reached} | mission_accomplished: {self._goal}>")

    def __repr__(self):
        return self.__str__()

    @property
    def xyz(self):
        return self._position[0:3]

    @property
    def terminal(self):
        return self._terminal

    @property
    def is_goal(self):
        return self._goal

cdef class Nav3DAction(Action):

    EPSILON_LONGI = np.pi/12
    EPSILON_LATI = np.pi/12
    cdef public float _longitude
    cdef public float _latitude

    def __init__(self, longitude, latitude):

        if longitude > np.pi or longitude < -np.pi:
            raise(ValueError, "Make sure longitude is in radians and in the principal range [-pi, pi].")

        if latitude > np.pi/2 or latitude < -np.pi/2:
            raise(ValueError, "Make sure latitude is in radians and in [-pi/2, pi/2].")

        self._longitude = longitude
        self._latitude = latitude

    def __hash__(self):
        # Keep this set to 1 if you want to identify actions via __eq__.
        # Python dictionaries identify keys if (__hash__ and __eq__) is True.
        return 1

    def __eq__(self, other):
        if not isinstance(other, Action):
            return False
        return (angle_distance(self._latitude, other._latitude) <= self.EPSILON_LATI
                and angle_distance(self._longitude, other._longitude) <= self.EPSILON_LONGI)

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

    def __eq__(self, Nav3DMacroAction other):
        if isinstance(other, Nav3DMacroAction):
            # if np.linalg.norm(self.effect() - other.effect()) < self.EPSILON * min(len(self._action_sequence), len(other._action_sequence)):
            if self.angle_between(self.effect(), other.effect()) < self.EPSILON_ANGLE and np.linalg.norm(self.effect() - other.effect()) <= self.EPSILON_EFFECT:
                return True
            return False
        return False

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

    EPSILON=1.0
    cdef public tuple _pos_reading

    def __init__(self, pos_reading):
        """
        pos_reading: The position reading of the observation.
        """
        self._pos_reading = pos_reading

    def __hash__(self):
        # Keep this set to 1 if you want to identify observations via __eq__.
        # Python dictionaries identify keys if (__hash__ and __eq__) is True.
        return 1

    def __eq__(self, other):
        if isinstance(other, Nav3DObservation):
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
    LEN_TOLERANCE = 1
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
        return False

    def __str__(self):
        return f"<obs_sequence: {[o._pos_reading for o in self.observation_sequence]}>"

    def __repr__(self):
        return self.__str__()

    @property
    def observation_sequence(self):
        return self._observation_sequence

cdef class Nav3DTransitionModel(TransitionModel):

    STEP_SIZE = 2.
    MEAN = list(np.array([0, 0, 0]))
    COV = list(np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]]) * 0.25 * STEP_SIZE)

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
        cdef list remaining_targets
        cdef list objectives_reached

        dv = self.STEP_SIZE * np.array([np.cos(action._latitude) * np.cos(action._longitude),
                                       np.cos(action._latitude) * np.sin(action._longitude),
                                       np.sin(action._latitude)]) + self.CONTROL_ERROR_DIST.random()

        next_position = tuple(np.array(state._position) + dv)

        # TODO: Rewrite without the deepcopy.
        objectives_reached = copy.deepcopy(state._objectives_reached)
        remaining_targets = [target for target in self._pyb_env.ordered_goal_configs if target not in state._objectives_reached]
        if len(remaining_targets):
            for target in remaining_targets:
                if (self._pyb_env.goal_checker(next_position + (0,0,0)) and np.linalg.norm(np.array(next_position+(0,0,0)) - np.array(target)) < 5):
                    objectives_reached += [target]

        return Nav3DState(position=next_position,
                          collision=self._pyb_env.collision_checker(next_position + (0, 0, 0)),
                          no_fly_zone=self._pyb_env.dz_checker(next_position + (0, 0, 0)),
                          landmark=self._pyb_env.lm_checker(next_position + (0, 0, 0)),
                          objectives_reached=objectives_reached,
                          goal=set(objectives_reached)==set(self._pyb_env.ordered_goal_configs))

cdef class Nav3DObservationModel(ObservationModel):

    MEAN = list(np.array([0, 0, 0]))
    COV = list(np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]]) * .2)

    OBS_ERROR_DIST = pomdp_py.Gaussian(mean=MEAN, cov=COV)

    cdef public PyBulletEnv _pyb_env
    def __init__(self, pyb_env):
        """
        pyb_env (PyBulletEnv): A PyBullet environment instance as defined in examples.motion_planning.environments.
        """
        self._pyb_env = pyb_env

    def probability(self, Nav3DObservation observation, Nav3DState next_state, Nav3DAction action):

        # if self._pyb_env.lm_checker(next_state._position + (0, 0, 0)):
        #     if observation._pos_reading == next_state._position:
        #         return 1.
        #     else:
        #         return 0.
        # else:
        #     return self.CONTROL_ERROR_DIST[np.array(observation._pos_reading) - np.array(next_state._position)]
        return self.OBS_ERROR_DIST[np.array(observation._pos_reading) - np.array(next_state._position)]

    def sample(self, Nav3DState next_state, Nav3DAction action):

        # if self._pyb_env.lm_checker(next_state._position + (0,0,0)):
        #     return Nav3DObservation(next_state._position)

        noisy_reading = self.OBS_ERROR_DIST.random()

        return Nav3DObservation(pos_reading=tuple(np.array(next_state._position) + np.array(noisy_reading)))

cdef class Nav3DRewardModel(RewardModel):

    cdef public PyBulletEnv _pyb_env

    STEP_REWARD = -2.5 # The step penalty for non-terminal states.
    NFZ_REWARD = -20 # The no fly zone penalty.
    COLLISION_REWARD = -2000 # The collision penalty.
    OBJECTIVE_REWARD = 2000 # The reward for each objective attained (in order).
    MISSION_ACCOMPLISHED_REWARD = 20000 # The reward for accomplishing the mission.

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

        return (self.STEP_REWARD
                + (self.COLLISION_REWARD if next_state._collision else 0)
                + (self.NFZ_REWARD if next_state._no_fly_zone else 0)
                + (self.OBJECTIVE_REWARD if len(next_state._objectives_reached) > len(state._objectives_reached) else 0)
                + (self.MISSION_ACCOMPLISHED_REWARD if next_state.is_goal else 0))

cdef class Nav3DBlackboxModel(BlackboxModel):

    cdef public PyBulletEnv _pyb_env
    cdef public Nav3DTransitionModel _Tm
    cdef public Nav3DObservationModel _Om
    cdef public Nav3DRewardModel _Rm

    def __init__(self, pyb_env):

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

    REACHED_DIST = 2.  # Distance at which the target is considered to be reached.

    def __init__(self, pyb_env: PyBulletEnv, max_macro_action_length=1,
                 prm_nodes=500,
                 prm_lower_bounds=(-100, -100, 0),
                 prm_upper_bounds=(100, 100, 50)):

        key_positions = list(map(lambda c : c[0:3], pyb_env.key_configs + pyb_env.goal_configs))

        self._pyb_env = pyb_env
        self._Tm = Nav3DTransitionModel(pyb_env)
        self._Rm = Nav3DRewardModel(pyb_env)
        self._prm = PRM(pyb_env,
                            3,
                            prm_lower_bounds,
                            prm_upper_bounds,
                            prm_nodes,
                            15,
                            sampling_dist="Uniform",
                            gaussian_noise=10,
                            nodes_to_include=key_positions)
        self._max_macro_action_length = max_macro_action_length
        self._discrete_actions = self.enumerate_discrete_actions()

    """
    Here the objective is an (ordered) list of waypoint and target and their 
    corresponding 'reached' statuses.
    """

    def new_global_objective(self, Nav3DState state):
        target = self.get_new_target(state)
        targets = [self.get_new_waypoint(state, target), target]
        kwargs = {"targets_reached": [False, False]}
        return Objective(states=targets, kwargs=kwargs)

    def maintain_global_objective(self, Nav3DState state, Objective objective):
        """If the objective has been attained, sample a new one.
        Otherwise, maintain the objective by keeping track of the statuses."""

        if objective is None or all(objective.kwargs.get("targets_reached")):
            return self.new_global_objective(state)

        return Objective(states=objective.states,
                         kwargs={"targets_reached": [self.target_reached(state, objective.states[0]),
                                                     self.target_reached(state, objective.states[1])]})

    def sample(self, Nav3DState state, Objective objective):
        """Find a collision-free trajectory to the last unattained state on
        the PRM and fashion a macro action that traces the top of the trajectory."""

        target = objective.states[1] if objective.kwargs.get("targets_reached")[0] else objective.states[0]
        path = self._prm.shortest_path_offline(state._position+(0,0,0), target)

        if len(path) < 2:
            # print("PRM path too short. "
            #       + "Taking the straight line distance irrespective of obstacles.")
            # NB: this just takes the straight line irrespective of obstacles.
            edges = [(state._position + (0,0,0), target)]
        else:
            edges = [(path[i], path[i+1]) for i in range(len(path) - 1)]
            # edges = [(path[0], path[1])]

        cdef list action_sequence = []

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

        # e = edges[0]
        # edge_vector = np.array(e[1]) - np.array(e[0])
        # edge_distance = np.linalg.norm(edge_vector)
        # action = Nav3DAction(np.arctan2(edge_vector[1], edge_vector[0]), np.arcsin(edge_vector[2]/edge_distance))
        # m = int(np.ceil(edge_distance / self._Tm.STEP_SIZE))
        # for j in range(m):
        #     action_sequence.append(action)
        #     if len(action_sequence) >= self._max_macro_action_length:
        #         return Nav3DMacroAction(action_sequence)

        return Nav3DMacroAction(action_sequence)

    def value_heuristic(self, Nav3DState state, Nav3DState next_state, float reward, float discount):

        if next_state is not None:
            if next_state.terminal:
                return reward

        d = int(self.distance_to_goal(state))
        return (self._Rm.STEP_REWARD * (1 - discount**d)/(1 - discount)) + self._Rm.OBJECTIVE_REWARD * (discount ** d) * 0.1

    def target_reached(self, Nav3DState state, tuple target):
        return np.linalg.norm(np.array(state._position + (0,0,0)) - np.array(
                target)) < self.REACHED_DIST

    def get_new_target(self, Nav3DState state):
        """Sample key configs, removing objectives that have already been achieved."""
        unvisited_objectives = set(self._pyb_env.key_configs) - set(state._objectives_reached)
        if len(unvisited_objectives):
            return list(unvisited_objectives)[np.random.randint(len(unvisited_objectives))]
        else:
            return list(self._pyb_env.goal_configs)[np.random.randint(len(self._pyb_env.goal_configs))]

    def get_uniform_ball_noise(self, max_radius):
        """Adds uniform noise in a ball centered at the origin."""
        lower_bounds = [0, 0, -np.pi/2]
        upper_bounds = [max_radius, 2*np.pi, np.pi/2]
        r, longi, lati = np.random.uniform(low=lower_bounds, high=upper_bounds, size=(1, 3))[0]
        x = r * np.cos(lati) * np.cos(longi)
        y = r * np.cos(lati) * np.sin(longi)
        z = r * np.cos(lati)
        return x, y, z, 0, 0, 0

    def get_uniform_box_noise(self, half_extent):
        """Adds uniform noise in a box centered at the origin."""
        lower_bounds = [-half_extent, -half_extent, -half_extent]
        upper_bounds = [half_extent, half_extent, half_extent]
        x, y, z = np.random.uniform(low=lower_bounds, high=upper_bounds, size=(1, 3))[0]
        return x, y, z, 0, 0, 0

    def get_midpoint(self, path):
        """Get the middle of the path from state and target."""
        midpoint = path[int(len(path) * .5)]
        return midpoint

    def get_new_waypoint(self, Nav3DState state, tuple target):
        """Construct the shortest collision-free path to the current target.
        Compute the distance r between the midpoint on this path to the target and
        sample a point inside the ball centered at the midpoint with radius r.
        Since this point may not be collision-free, query the nearest collision-free
        point on the PRM and return this as the desired waypoint."""

        if np.random.random() <= 0.5:
            # orig_path = self._prm.shortest_path_offline(state._position+(0,0,0), target)
            # midpoint = self.get_midpoint(orig_path) if len(orig_path)>2 else target
            # waypoint_distance = np.linalg.norm(np.array(midpoint) - np.array(target))
            midpoint = tuple(0.5 * (np.array(state._position + (0,0,0)) + 0.5 * np.array(target)))
            midpoint_distance = np.linalg.norm(np.array(midpoint) - np.array(target))
            # TODO: Hardcoded scaling factor.
            waypoint = tuple(np.array(midpoint) + np.array(self.get_uniform_ball_noise(midpoint_distance * 1)))
        else:
            waypoint = self.get_new_target(state)

        # waypoint = tuple(np.array(midpoint) + np.array(self.get_uniform_box_noise(midpoint_distance * 1.25)))
        nearest_d, nearest_i = self._prm.kd_tree.query(waypoint)
        return tuple(self._prm.kd_tree.data[nearest_i])

    # def get_new_waypoint(self, Nav3DState state, tuple target, lower_bounds=(-100, -100, 0), upper_bounds=(100, 100, 70)):
    #     """
    #     A simpler way to generate homotopic paths by random sampling.
    #     """
    #     if np.random.random() <= 0.5: # TODO: Hardcoded.
    #         waypoint = tuple(np.random.uniform(low=lower_bounds, high=upper_bounds)) + (0., 0., 0.)
    #     else:
    #         waypoint = self.get_new_target(state)
    #     nearest_d, nearest_i = self._prm.kd_tree.query(waypoint)
    #     return tuple(self._prm.kd_tree.data[nearest_i])

    def distance_to_goal(self, Nav3DState state):

        unvisited_goals = set(self._pyb_env.goal_configs) - set(state._objectives_reached)

        if len(unvisited_goals):

            dist = np.inf
            for goal in unvisited_goals:
                dist_to_goal = self._prm.shortest_path_length(np.array(state._position + (0, 0, 0)),
                                                              np.array(goal)) / self._Tm.STEP_SIZE
                if dist_to_goal < dist:
                    dist = dist_to_goal

            return dist
        else:
            return 0

    """The remaining functions are for a uniform rollout for POMCP."""

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

class HolonomicRescuePOMDP(MPPOMDP):

    def __init__(self,
                 init_state,
                 init_belief,
                 pyb_env,
                 pyb_env_gui,
                 max_macro_action_length=1,
                 prm_nodes=500,
                 prm_lower_bounds=(-100, -100, 0),
                 prm_upper_bounds=(100, 100, 100)):
        self._init_state = init_state
        self._init_belief = init_belief,
        self._pyb_env = pyb_env
        self._pyb_env_gui = pyb_env_gui

        "Agent"
        agent = Agent(init_belief=init_belief,
                             policy_model=Nav3DMacroReferencePolicyModel(pyb_env=pyb_env,
                                                                      max_macro_action_length=max_macro_action_length,
                                                                      prm_nodes=prm_nodes,
                                                                      prm_lower_bounds=prm_lower_bounds,
                                                                      prm_upper_bounds=prm_upper_bounds),
                             transition_model=Nav3DTransitionModel(pyb_env=pyb_env),
                             observation_model=Nav3DObservationModel(pyb_env=pyb_env),
                             reward_model=Nav3DRewardModel(pyb_env=pyb_env),
                             blackbox_model=Nav3DBlackboxModel(pyb_env=pyb_env),
                             name=f"Holonomic3DAgent({max_macro_action_length})")

        "Environment"
        env = Environment(
            init_state=init_state,
            transition_model=Nav3DTransitionModel(pyb_env=pyb_env_gui),
            reward_model=Nav3DRewardModel(pyb_env=pyb_env_gui))

        super().__init__(agent, env, name="Holonomic3DPOMDP")

    def visualize_belief(self, histogram, timeout=10., life_time=.1, rgb=(1,0,1)):
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

    def visualize_world(self, track_robot=False):
        self._pyb_env_gui.set_config(self.env.state._position + (0,0,0))
        if track_robot:
            pyb.resetDebugVisualizerCamera(cameraDistance=50,
                                           cameraYaw=-90,
                                           cameraPitch=-20,
                                           cameraTargetPosition=self.env.state._position,
                                           physicsClientId=self._pyb_env_gui._id)

    def update_environment(self, env_index=0):

        if self._pyb_env._id == pyb.GUI:
            pyb.disconnect(self._pyb_env._id)
            pyb_env_gui = Corsica(gui=True, debugger=False, dz_level=env_index, track_robot=False)
            self._pyb_env = pyb_env_gui
            self._pyb_env_gui = pyb_env_gui
        else:
            pyb.disconnect(self._pyb_env._id)
            pyb_env = Corsica(gui=False, debugger=False, dz_level=env_index, track_robot=False)
            pyb.disconnect(self._pyb_env_gui._id)
            pyb_env_gui = Corsica(gui=False, debugger=False, dz_level=env_index, track_robot=False)
            # pyb_env_gui = Corsica(gui=True, debugger=False, dz_level=env_index, track_robot=False)
            self._pyb_env = pyb_env
            self._pyb_env_gui = pyb_env_gui

        self.agent.transition_model = Nav3DTransitionModel(self._pyb_env)
        self.agent.observation_model = Nav3DObservationModel(self._pyb_env)
        self.agent.reward_model = Nav3DRewardModel(self._pyb_env)
        self.agent.blackbox_model = Nav3DBlackboxModel(self._pyb_env)

        cur_state = copy.deepcopy(self.env.cur_state)

        self.env = Environment(
            init_state=cur_state,
            transition_model=Nav3DTransitionModel(pyb_env=self._pyb_env_gui),
            reward_model=Nav3DRewardModel(pyb_env=self._pyb_env_gui))

    def reset_environment(self):

        env_index = 0

        if self._pyb_env._id == pyb.GUI:
            pyb.disconnect(self._pyb_env._id)
            pyb_env_gui = Corsica(gui=True, debugger=False, dz_level=env_index, track_robot=False)
            self._pyb_env = pyb_env_gui
            self._pyb_env_gui = pyb_env_gui
        else:
            pyb.disconnect(self._pyb_env._id)
            pyb_env = Corsica(gui=False, debugger=False, dz_level=env_index, track_robot=False)
            pyb.disconnect(self._pyb_env_gui._id)
            pyb_env_gui = Corsica(gui=False, debugger=False, dz_level=env_index, track_robot=False)
            # pyb_env_gui = Corsica(gui=True, debugger=False, dz_level=env_index, track_robot=False)
            self._pyb_env = pyb_env
            self._pyb_env_gui = pyb_env_gui

        self.agent.transition_model = Nav3DTransitionModel(self._pyb_env)
        self.agent.observation_model = Nav3DObservationModel(self._pyb_env)
        self.agent.reward_model = Nav3DRewardModel(self._pyb_env)
        self.agent.blackbox_model = Nav3DBlackboxModel(self._pyb_env)

        self.env = Environment(
            init_state=self._init_state,
            transition_model=Nav3DTransitionModel(pyb_env=self._pyb_env_gui),
            reward_model=Nav3DRewardModel(pyb_env=self._pyb_env_gui))




