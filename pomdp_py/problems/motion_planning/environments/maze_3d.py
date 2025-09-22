from pomdp_py.problems.motion_planning.environments.pyb_env import *
from pomdp_py.problems.motion_planning import pyb_utils

import numpy as np
import pybullet as pyb
import time

np.set_printoptions(precision=3, suppress=True)  # for neat printing of numpy arrays.

TIMESTEP = 1.0/600
class Maze3D(PyBulletEnv):

    LANDMARKS = [
            (-26, -20, -4),
            (-20, -25, -4),
            (2, -9, -4),
            (7, 8, 0),
            (17, -7, 6),
            (27, 10, -4)
        ]

    GOAL_CONFIGS = [
        (17, -7, 6),
        (27, 10, -4)
    ]

    KEY_CONFIGS = LANDMARKS + GOAL_CONFIGS * 5

    def __init__(self,
                 robot_init_config=(.0, .0, .0, .0, .0, .0),
                 gui=False,
                 debugger=False,
                 collision_margin=0.1):

        # Use the GUI for the execution environment and DIRECT for planning environment.
        self._id = pyb.connect(pyb.GUI) if gui else pyb.connect(pyb.DIRECT)

        if not debugger:
            pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 0)

        pyb.resetDebugVisualizerCamera(cameraDistance=55,
                                       cameraYaw=0,
                                       cameraPitch=-80,
                                       cameraTargetPosition=[0, -0, 0],
                                       physicsClientId=self._id)

        self._robot_init_config = robot_init_config
        self._collision_margin = collision_margin

        # Load objects.
        self._robot = self.load_robot()
        self._goals = self.load_goals()
        self._landmarks = self.load_landmarks()
        self._danger_zones = self.load_danger_zones()
        self._collision_objects = self.load_collision_objects()

        #########################################################
        # COLLISION CHECKING
        #########################################################
        self._col_detector = pyb_utils.CollisionDetector(
            self._id,
            [(self._robot, _) for _ in self._collision_objects]
        )

        #########################################################
        # DANGER ZONE CHECKING
        #########################################################
        self._dz_detector = pyb_utils.CollisionDetector(
            self._id,
            [(self._robot, _) for _ in self._danger_zones]
        )

        #########################################################
        # LANDMARK CHECKING
        #########################################################
        # IMPORTANT! We assert that zones are also landmarks
        # so that the agent receives feedback if it has entered
        # a danger zone.
        self._lm_detector = pyb_utils.CollisionDetector(
            self._id,
            [(self._robot, _) for _ in self._landmarks + self._danger_zones]
        )


        #########################################################
        # GOAL CHECKING
        #########################################################

        self._goal_detector = pyb_utils.CollisionDetector(
            self._id,
            [(self._robot, _) for _ in self._goals]
        )

    def load_robot(self):

        boxes = [
            [[0.5, 0.5, 0.5], self._robot_init_config[0:3], [0, 0, 0]]
        ]

        return self.load_primitive_objects(boxes=boxes, rgba=[3, 0.1, 0.1, 1])[0]

    def load_goals(self, visualize=True):

        spheres = [
            [3, [17, -7, 6]],
            [3, [27, 10, -4]]
        ]

        return self.load_primitive_objects(spheres=spheres, rgba=[0, 1, 1, 1])

    def load_landmarks(self):

        boxes = [
            [[2, 2, 2], [-26, -20, -5], [0, 0, 0]],
            [[2, 2, 2], [-20, -25, -5], [0, 0, 0]],
            [[2, 2, 2], [2, -9, -5], [0, 0, 0]],
            [[2, 2, 2], [7, 8, 0], [0, 0, 0]],
            [[3, 3, 3], [17, -7, 6], [0, 0, 0]],
            [[3, 3, 3], [27, 10, -4], [0, 0, 0]]
        ]

        return self.load_primitive_objects(boxes=boxes, rgba=[1, 0.1, 1, 0.5])

    def load_danger_zones(self):

        # boxes = [
        #     [[10, 1, 6], [-11, 17, 2.0], [0, 0, 0]],
        #     [[10, 1, 4], [-11, 13, 2.0], [0, 0, 0]],
        #     [[5, 1, 4], [-5, -26, 2.0], [0, 0, 0]]
        # ]

        boxes = [
            [[10, 1, 10], [-11, 17, 2.0], [0, 0, 0]],
            [[10, 1, 10], [-11, 13, 2.0], [0, 0, 0]],
            [[5, 2, 6], [-5, -27, -3.0], [0, 0, 0]]
        ]

        return self.load_primitive_objects(boxes=boxes, rgba=[1, 1, 0.2, 0.8])

    def load_collision_objects(self):

        heightfields = [
                # [Path(__file__).parent.parent.parent.parent / "data/terrain.png",
                #  (1. / 256 * 50., 1. / 256 * 50., 1 / 0.5)]
            ]

        spheres = [
            # [1, [1, 1, 1]]
        ]

        capsules = [
            # [1, 1, [5, 5, 0], [0, 0, 0]],
            # [0.5, 5, [1, 2, 1], [1, 2, 3]]
        ]

        boxes = [
            [[30, 30, 5], [0.0, 0.0, -15.0], [0, 0, 0]],
            [[30, 30, 5], [0.0, 0.0, 15.0], [0, 0, 0]],
            [[2, 32, 20], [32, 0, 0], [0, 0, 0]],
            [[2, 32, 20], [-32, 0, 0], [0, 0, 0]],
            [[32, 2, 20], [0, 32, 0], [0, 0, 0]],
            [[32, 2, 20], [0, -32, 0], [0, 0, 0]],
            [[1, 5, 10], [-23, 25, 0], [0, 0, 0]],
            [[1, 9, 10], [-23, 2, 0], [0, 0, 0]],
            [[1, 5, 10], [-23, -19, 0], [0, 0, 0]],
            [[1, 14, 10], [12, -16, 0], [0, 0, 0]],
            [[1, 10, 10], [12, 20, 0], [0, 0, 0]],
            [[10, 1, 10], [20, 5, 0], [0, 0, 0]],
            [[14, 1, 10], [-2, 20, 0], [0, 0, 0]],
            [[13.5, 1, 10], [-10.5, -6, 0], [0, 0, 0]],
            [[1, 9, 10], [2, 2, 0], [0, 0, 0]],
            [[13.5, 1, 10], [-10.5, -15, 0], [0, 0, 0]],
            [[1, 5, 10], [2, -19, 0], [0, 0, 0]],
            [[8, 1, 10], [-5, -23, 0], [0, 0, 0]],
            [[5, 1, 10], [16, -3, 0], [0, 0, 0]]
        ]

        return self.load_primitive_objects(spheres=spheres,
                                            capsules=capsules,
                                            boxes=boxes,
                                            heightfields=heightfields)


    def load_primitive_objects(self, **kwargs):

        spheres = kwargs.get("spheres", [])
        capsules = kwargs.get("capsules", [])
        boxes = kwargs.get("boxes", [])
        heightfields = kwargs.get("heightfields", [])
        rgba = kwargs.get("rgba", [1, 1, 1, 0.2])

        objects = []

        for radius, center in spheres:
            objects.append(pyb.createMultiBody(0,
                 pyb.createCollisionShape(pyb.GEOM_SPHERE,
                                           radius=radius,
                                           physicsClientId=self._id),
                 basePosition=center,
                 physicsClientId=self._id))


        for radius, height, center, euler in capsules:
            objects.append(pyb.createMultiBody(0,
                 pyb.createCollisionShape(pyb.GEOM_CAPSULE,
                                           radius=radius,
                                           height=height,
                                           physicsClientId=self._id),
                 basePosition=center,
                 baseOrientation=pyb.getQuaternionFromEuler(euler),
                 physicsClientId=self._id))


        for half_extents, center, euler in boxes:
            objects.append(pyb.createMultiBody(0,
                   pyb.createCollisionShape(pyb.GEOM_BOX,
                                            halfExtents=half_extents,
                                            physicsClientId=self._id),
                   basePosition=center,
                   baseOrientation=pyb.getQuaternionFromEuler(euler),
                   physicsClientId=self._id))

        for file_path, mesh_scale in heightfields:

            shape = pyb.createCollisionShape(
                shapeType=pyb.GEOM_HEIGHTFIELD,
                meshScale=mesh_scale,
                fileName=str(file_path),
                physicsClientId=self._id
            )

            shape_id = pyb.createMultiBody(0, shape, physicsClientId=self._id)
            objects.append(shape_id)

        for obj_id in objects:
            pyb.changeVisualShape(objectUniqueId=obj_id, linkIndex=-1, rgbaColor=rgba, physicsClientId=self._id)

        return objects

    def set_config(self, config):
        pos = config[0:3]
        ori = config[3:6]

        q = quaternion_from_euler(*ori)

        pyb.resetBasePositionAndOrientation(
            self._robot, pos, q, physicsClientId=self._id
        )

    def reset(self):
        """Resets the environment to its initial state."""

        self.set_config(self._robot_init_config)

    def goal_checker(self, config):

        self.set_config(config)

        return self._goal_detector.in_collision(margin=self._collision_margin)

    def lm_checker(self, config):

        self.set_config(config)

        return self._lm_detector.in_collision(margin=self._collision_margin)

    def dz_checker(self, config):

        self.set_config(config)

        return self._dz_detector.in_collision(margin=self._collision_margin)

    def collision_checker(self, config):

        self.set_config(config)

        return self._col_detector.in_collision(margin=self._collision_margin)

    def sample_key_config(self):

        return tuple(self.KEY_CONFIGS[np.random.randint(len(self.KEY_CONFIGS))]) + (0, 0, 0)

    def sample_goal_config(self):

        return tuple(self.GOAL_CONFIGS[np.random.randint(len(self.GOAL_CONFIGS))]) + (0, 0, 0)

    def default_goal_config(self):

        return (17, -7, 6) + (0, 0, 0)

    @property
    def goal_configs(self):

        return [(27, 10, -4) + (0, 0, 0), (17, -7, 6) + (0, 0, 0)]

def main():

    env_gui = Maze3D(gui=True, debugger=True)

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