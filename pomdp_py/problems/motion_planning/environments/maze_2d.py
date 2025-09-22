# TODO: Remove reference to _robot.uid and client_id parameter from load_environment (see Maze3D).

from pomdp_py.problems.motion_planning.environments.pyb_env import *
from pomdp_py.problems.motion_planning import pyb_utils

import numpy as np
import pybullet as pyb
import pybullet_data
import time
import pathlib

np.set_printoptions(precision=3, suppress=True)  # for neat printing of numpy arrays.

TIMESTEP = 1.0/60
class Maze2D(PyBulletEnv):

    def __init__(self,
                 robot_init_config=(0., 0., .0, .0, .0, .0),
                 collision_margin=0.01,
                 gui=False,
                 debugger=False):

        self._robot_init_config = robot_init_config
        self._collision_margin = collision_margin

        # Use the GUI for the execution environment and DIRECT for planning environment.
        self._id = pyb.connect(pyb.GUI) if gui else pyb.connect(pyb.DIRECT)

        if not debugger:
            pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 0)

        # Uncomment if you want the GUI camera view to load at default position.
        pyb.resetDebugVisualizerCamera(cameraDistance=50,
                                       cameraYaw=0,
                                       cameraPitch=-80,
                                       cameraTargetPosition=[30, -30, 0],
                                       physicsClientId=self._id)

        self._robot, self._obstacles, self._danger_zones, self._landmarks, self._goals = self.load_environment(self._id)

        #########################################################
        # STATIC DATA
        #########################################################

        self._goal_positions = [(49, -37), (58, -22)]
        self._key_positions = [(4, -51), (4, -60), (10, -58), (37, -25), (31, -58), (31, -40),
                               (37, -25), (40, -60), (43, -32)] # landmarks sans goals.

        #########################################################
        # COLLISION CHECKING
        #########################################################
        self._col_detector = pyb_utils.CollisionDetector(
            self._id,
            [(self._robot.uid, _) for _ in self._obstacles]
        )

        #########################################################
        # DANGER ZONE CHECKING
        #########################################################
        self._dz_detector = pyb_utils.CollisionDetector(
            self._id,
            [(self._robot.uid, _) for _ in self._danger_zones]
        )

        #########################################################
        # LANDMARK CHECKING
        #########################################################
        self._lm_detector = pyb_utils.CollisionDetector(
            self._id,
            [(self._robot.uid, _) for _ in self._landmarks + self._danger_zones]
        )


        #########################################################
        # GOAL CHECKING
        #########################################################

        self._goal_detector = pyb_utils.CollisionDetector(
            self._id,
            [(self._robot.uid, _) for _ in self._goals]
        )

    def load_environment(self, client_id):

        pyb.setTimeStep(TIMESTEP, physicsClientId=client_id)
        pyb.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=client_id
        )

        # Helper functions for importing objects
        def load_box_at_XY(xy_coord: tuple, z_coord=0.0, rgba=(0, 0, 0, .5)):
            base_position = (xy_coord[0], -xy_coord[1], z_coord)
            obj_id = pyb.loadURDF(
                str((pathlib.Path(__file__).parent.parent.parent.parent / "data/cuboid.urdf")),
                base_position, useFixedBase=True, physicsClientId=client_id
            )
            pyb.changeVisualShape(obj_id, -1, rgbaColor=rgba)
            return obj_id

        # robot
        robot_id = pyb.loadURDF(
            str((pathlib.Path(__file__).parent.parent.parent.parent / "data/cuboid.urdf")),
            self._robot_init_config[0:3],
            pyb.getQuaternionFromEuler(self._robot_init_config[3:6]),
            globalScaling=.8,
            physicsClientId=client_id
        )
        # pyb.changeVisualShape(robot_id, -1, rgbaColor=[0, 1, 1, 1.0])
        robot = pyb_utils.Robot(robot_id, client_id=client_id)

        # ground plane
        ground_id = pyb.loadURDF(
            "plane.urdf", [30.5, -30.5, -0.5], globalScaling=2., useFixedBase=True, physicsClientId=client_id
        )
        pyb.changeVisualShape(ground_id, -1, rgbaColor=[1, 1, 1, 1])

        # obstacles
        obstacles_id = pyb.loadURDF(
            str((pathlib.Path(__file__).parent.parent.parent.parent / "data/gridmap/obstacles.urdf")),
            basePosition=[0, 0, 0], baseOrientation=pyb.getQuaternionFromEuler([0, 0, np.pi/2]), useFixedBase=True, physicsClientId=client_id
        )

        # danger zones
        danger_zones_id = pyb.loadURDF(
            str((pathlib.Path(__file__).parent.parent.parent.parent / "data/gridmap/danger_zones.urdf")),
            basePosition=[0, 0, 0], baseOrientation=pyb.getQuaternionFromEuler([0, 0, np.pi/2]), useFixedBase=True, physicsClientId=client_id
        )

        # landmarks
        landmarks_id = pyb.loadURDF(
            str((pathlib.Path(__file__).parent.parent.parent.parent / "data/gridmap/landmarks.urdf")),
            basePosition=[0, 0, 0], baseOrientation=pyb.getQuaternionFromEuler([0, 0, np.pi/2]), useFixedBase=True, physicsClientId=client_id
        )

        # goals
        goals_id = pyb.loadURDF(
            str((pathlib.Path(__file__).parent.parent.parent.parent / "data/gridmap/goals.urdf")),
            basePosition=[0, 0, 0], baseOrientation=pyb.getQuaternionFromEuler([0, 0, np.pi/2]), useFixedBase=True, physicsClientId=client_id
        )

        obstacles = [
            # ground_id,
            obstacles_id,
            # danger_zones_id
        ]

        danger_zones = [
            danger_zones_id
        ]

        landmarks = [
            landmarks_id
        ]

        goals = [
            goals_id
        ]

        return robot, obstacles, danger_zones, landmarks, goals

    def sample_key_config(self):

        # return (np.random.randint(60), -np.random.randint(60), 0, 0, 0, 0)

        positions = self._key_positions + self._goal_positions * 10
        return tuple(list(positions)[np.random.randint(len(positions))]) + (0, 0, 0, 0)

    def sample_goal_config(self):
        """Samples a tuple of (x, y, z, roll, pitch, yaw) for the goal of interest."""

        return tuple(list(self._goal_positions)[np.random.randint(len(self._goal_positions))]) + (0, 0, 0, 0)

    def default_goal_config(self):

        return self._goal_positions[0] + (0, 0, 0, 0)

    def sample_free_config(self):

        x = np.random.choice(range(60)) + 1
        y = -np.random.choice(range(60)) - 1
        pos = (x, y, 0)
        ori = (0, 0, 0)
        config = pos + ori

        while self.collision_checker(config):
            x = np.random.choice(range(60)) + 1
            y = -np.random.choice(range(60)) - 1
            pos = (x, y, 0)
            ori = (0, 0, 0)
            config = pos + ori

        return config

    def set_config(self, config):

        pos = config[0:3]
        ori = config[3:6]

        pyb.resetBasePositionAndOrientation(
            self._robot.uid, pos, pyb.getQuaternionFromEuler(ori), physicsClientId=self._robot.client_id
        )

    def collision_checker(self, config):
        """Takes a configuration and returns True if it is in collision with an obstacle."""

        self.set_config(config)

        return self._col_detector.in_collision(margin=self._collision_margin)

    def dz_checker(self, config):
        """Takes a configuration and returns True if it is at a danger_zone."""

        self.set_config(config)

        return self._dz_detector.in_collision(margin=self._collision_margin)

    def lm_checker(self, config):
        """Takes a configuration and returns True if it is at a danger_zone."""

        self.set_config(config)

        return self._lm_detector.in_collision(margin=self._collision_margin)

    def goal_checker(self, config):
        """Takes a configuration and returns True if it is at a danger_zone."""

        self.set_config(config)

        return self._goal_detector.in_collision(margin=self._collision_margin)

    def reset(self):
        """Resets the environment to its initial state."""

        self.set_config(self._robot_init_config)

    @property
    def goal_positions(self):
        return self._goal_positions


def main():

    _robot_init_config = [0, 0, 0, 0, 0, 0]

    env_gui = Maze2D(robot_init_config=_robot_init_config,
                     gui=True, debugger=True)

    # create user debug parameters
    robot_params_gui = create_robot_params_gui(env_gui._id)

    while True:
        config = read_robot_params_gui(robot_params_gui, client_id=env_gui._id)

        if env_gui.goal_checker(config):
            print("Goal!")
        elif env_gui.collision_checker(config):
            print("Collision!")
        elif env_gui.dz_checker(config):
            print("Danger Zone!")
        elif env_gui.lm_checker(config):
            print("Landmark!")
        else:
            print("Free.")

        time.sleep(TIMESTEP)

if __name__ == "__main__":
    main()