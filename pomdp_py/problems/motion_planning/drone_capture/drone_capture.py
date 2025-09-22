# TODO: Bespoke log_replayer for drone_capture experiment logs run by Yuanchu Liang for ISRR 2024.
from PIL import ImageGrab
from pomdp_py.framework import basics
from pomdp_py.framework.basics import sample_generative_model
from pomdp_py.algorithms.po_uct import RolloutPolicy
import pomdp_py.problems.motion_planning.environments as env
import pomdp_py.problems.motion_planning.path_planning as prm
import pomdp_py
import numpy as np
import random
import time
from scipy.stats import multivariate_normal
from pomdp_py.representations.distribution.particles import Particles
import pybullet as pyb

DIMENSION = 3
STEP_SIZE = 0.5
MOTIONS = {"N": (0, STEP_SIZE, 0),
           "S": (0, -STEP_SIZE, 0),
           "E": (STEP_SIZE, 0, 0),
           "W": (-STEP_SIZE, 0, 0),
           "U": (0, 0, STEP_SIZE),
           "D": (0, 0, -STEP_SIZE)}
OBS_NOISE = 0.25
OBS_DIST = multivariate_normal(mean=np.zeros(DIMENSION), cov=np.eye(DIMENSION)*OBS_NOISE)
PREDATOR_DETECTION_RADIUS = 6
PREY_DETECTION_RADIUS = 4

GOAL_REWARD = 500.
STEP_REWARD = -.1


def sup_norm(tuple1, tuple2):
    return np.max(abs(np.array(tuple1) - np.array(tuple2)))


def euclid_norm(tuple1, tuple2):
    return np.linalg.norm(np.array(tuple1) - np.array(tuple2))


def tuple_list_to_list(tuple_list):
    return [item for t in tuple_list for item in t]

def list_to_tuple_list(config_list):
    return [tuple(config_list[3*i:3*(i+1)]) for i in range(len(config_list) // 3)]

def tuple_round(tuple_3d, dps=0):
    return tuple(map(lambda c: round(c, dps), tuple_3d))

class PPState(basics.State):

    """
    Each state consists of:

    1. list of the `(x, y, z)` position of all predators
    2. list of the `(x, y, z)` position of all prey
    3. list of whether prey has been captured or not (`0=no`, `1=yes`)
    """

    def __init__(self,
                 predator_positions: list,
                 prey_positions: list,
                 captured: list):

        self._predator_positions = predator_positions
        self._prey_positions = prey_positions
        self._captured = captured
        self._terminal = self._captured

    @property
    def predator_positions(self):
        return self._predator_positions

    @property
    def prey_positions(self):
        return self._prey_positions

    @property
    def captured(self):
        return self._captured

    @property
    def terminal(self):
        return self._terminal

    @property
    def is_goal(self):
        return self.terminal

    def __hash__(self):
        return 1

    def __eq__(self, other):
        if not isinstance(other, PPState):
            return False
        else:
            return (np.all([self.predator_positions == other.predator_positions])
                    and np.all([self.prey_positions == other.prey_positions])
                    and np.all([self.captured == other.captured]))

    def __str__(self):
        return (f"<prey: {[tuple_round(pos, 1) for pos in self.prey_positions]} | "
                f"pred: {[tuple_round(pos, 1) for pos in self.predator_positions]} | "
                f"captured: {self.captured}>")

    def __repr__(self):
        return self.__str__()

class DroneCapturePOMDP(basics.MPPOMDP):

    def __init__(self,
                 init_state,
                 init_belief,
                 pyb_env,
                 pyb_env_gui):

        self._pyb_env = pyb_env
        self._pyb_env_gui = pyb_env_gui

        Tm = None
        Zm = None
        Rm = None
        Pm = None

        "Agent"
        agent = basics.Agent(init_belief=init_belief,
                             policy_model=Pm,
                             transition_model=Tm,
                             observation_model=Zm,
                             reward_model=Rm,
                             blackbox_model=None)

        "Environment"
        env = basics.Environment(
            init_state=init_state,
            transition_model=Tm,
            reward_model=Rm)

        print("Initializing Drone Capture Environment...")

        super().__init__(agent, env, name="DroneCapturePOMDP")

    def visualize_world(self):
        self._pyb_env_gui.set_config(tuple_list_to_list(self.env.state.predator_positions))
        self._pyb_env_gui.set_prey_config(tuple_list_to_list(self.env.state.prey_positions))

    def visualize_belief(self, histogram, step, timeout=10., life_time=.1, rgb=(0,0,1)):
        robots = []
        elapsed_time = 0.
        for s, prob in histogram.items():
            start_time = time.time()

            robot_id = pyb.createMultiBody(0,
                                           pyb.createCollisionShape(pyb.GEOM_SPHERE,
                                           radius=self._pyb_env.CAPTURE_RADIUS,
                                           physicsClientId=self._pyb_env_gui._id),
                                           basePosition=s._prey_positions,
                                           physicsClientId=self._pyb_env_gui._id)
            pyb.changeVisualShape(robot_id, -1, rgbaColor=rgb+(prob*.95 + 0.05,),
                                  physicsClientId=self._pyb_env_gui._id)
            robots.append(robot_id)

            end_time = time.time()
            elapsed_time += (end_time - start_time)
            if elapsed_time >= timeout:
                break

        # time.sleep(life_time)
        input("Press Enter to take screenshot...")
        screenshot = ImageGrab.grab()
        screenshot.save(f"drone_capture_step_{step}.png")
        screenshot.close()

        for robot_id in robots:
            pyb.changeVisualShape(robot_id, -1, rgbaColor=[0, 0, 0, 0], physicsClientId=self._pyb_env_gui._id)