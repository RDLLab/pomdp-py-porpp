from xml.sax.handler import property_dom_node

from pomdp_py.problems.motion_planning.environments.pyb_env import *
from pomdp_py.problems.motion_planning import pyb_utils

import numpy as np
import pybullet as pyb
import pybullet_data as pyb_data
import time
import pathlib

np.set_printoptions(precision=3, suppress=True)  # for neat printing of numpy arrays.

TIMESTEP = 1.0/600
class Corsica(PyBulletEnv):

    def __init__(self,
                 robot_init_config=(.0, .0, .0, .0, .0, .0),
                 gui=False,
                 debugger=False,
                 dz_level=0,
                 collision_margin=0.001,
                 camera_distance=130,
                 camera_yaw=-200,
                 camera_pitch=-45,
                 camera_target=(10, -50, 0),
                 track_robot=False):

        # Use the GUI for the execution environment and DIRECT for planning environment.
        self._id = pyb.connect(pyb.GUI) if gui else pyb.connect(pyb.DIRECT)

        if not debugger:
            pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 0)

        pyb.resetDebugVisualizerCamera(cameraDistance=camera_distance,
                                       cameraYaw=camera_yaw,
                                       cameraPitch=camera_pitch,
                                       cameraTargetPosition=camera_target,
                                       physicsClientId=self._id)

        self._robot_init_config = robot_init_config
        self._dz_level = dz_level
        self._collision_margin = collision_margin
        self._track_robot = track_robot

        # Load objects.
        self._robot = self.load_robot()
        self._goals = self.load_goals()
        self._landmarks = self.load_landmarks()
        self._danger_zones = self.load_danger_zones()
        self._collision_objects = self.load_collision_objects()
        self._terrain_collision_objects = self.load_terrain_collision_objects()

        #########################################################
        # COLLISION CHECKING
        #########################################################
        self._col_detector = pyb_utils.CollisionDetector(
            self._id,
            [(self._robot, _) for _ in self._collision_objects + self._terrain_collision_objects]
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

        # Define key data.
        self._key_configs = [
            (0, 0, 35, 0, 0, 0),
            (100, -40, 20, 0, 0, 0)
        ]

        self._goal_configs = [
            (0, 0, 35, 0, 0, 0),
            (100, -40, 20, 0, 0, 0)
        ]

    @property
    def target_1_config(self):
        return self._goal_configs[0]

    @property
    def target_2_config(self):
        return self._key_configs[1]

    @property
    def ordered_goal_configs(self):
        return self._goal_configs

    def load_robot(self):

        urdf_files = [
            # [pathlib.Path(__file__).parent.parent.parent.parent / "data/quadrotor.urdf", [0, 0, 0], 2]
        ]

        boxes = [
            [[1.5, .5, .25], [0, 0, 0], [0, 0, 0]]
        ]

        return self.load_primitive_objects(boxes=boxes, rgba=[2.0, .23, .13, 1])[0]

    def load_goals(self, visualize=True):

        spheres = [
            [4, [0, 0, 35]],
            [4, [100, -40, 20]]
        ]

        return self.load_primitive_objects(spheres=spheres, rgba=[.15, 1, .15, .5])

    def load_landmarks(self):

        boxes = [
            # [[2, 2, 2], [-26, -20, 5], [0, 0, 0]],
        ]

        return self.load_primitive_objects(boxes=boxes, rgba=[1, 0.1, 1, 0.5])

    def load_danger_zones(self):

        if self._dz_level == 0:
            boxes = []
        elif self._dz_level == 1:
            boxes = [
                [[10, 10, 10], [-45, 0, 20.], [0, 0, 0]]
            ]
        elif self._dz_level == 2:
            boxes = [
                [[60, 40, 20], [0, 30, 55.], [0, 0, 0]]
            ]
        elif self._dz_level == 3:
            boxes = [
                [[60, 40, 10], [0, 0, 42.], [0, 0, 0]],
                [[5, 20, 2], [90, -40, 30], [0, 0, 0]]
            ]

        return self.load_primitive_objects(boxes=boxes, rgba=[1, 0.2, 0.2, 0.3])

    def load_terrain_collision_objects(self):

        heightfields = [
                # [pathlib.Path(__file__).parent.parent.parent.parent / "data/terrain.png",
                #  (1. / 256 * 100., 1. / 256 * 100., 1 / 0.05),
                #  (-83600, -380300, 0)]
            ]

        scale = 4.
        meshes = [
            [pathlib.Path(__file__).parent.parent.parent.parent / "data/corsica.obj",
             (1. / 12.5 / scale, 1. / 12.5 / scale, 2. / 12.5 / scale),
             (-83600 / scale, -380300 / scale, 0)
             ]
        ]

        urdf_files = [
            # [pathlib.Path(__file__).parent.parent.parent.parent / "data/sydney.urdf", [-168300, 37730, 0], .01]
        ]

        return self.load_primitive_objects(heightfields=heightfields,
                                           meshes=meshes,
                                           urdf_files=urdf_files,
                                           rgba=[1.96/2, 1.64/2, 1.32/2, 1])

    def load_collision_objects(self):

        spheres = [
            # [1, [0, 0, 0]]
        ]

        capsules = [
            # [1, 1, [5, 5, 0], [0, 0, 0]],
        ]

        boxes = [
            [[300, 300, .5], [0.0, 0.0, 0], [0, 0, 0]]
        ]

        return self.load_primitive_objects(spheres=spheres,
                                           capsules=capsules,
                                           boxes=boxes,
                                           rgba=[.3/3., 1.38/3., 2.55/3., 1.])


    def load_primitive_objects(self, **kwargs):

        spheres = kwargs.get("spheres", [])
        capsules = kwargs.get("capsules", [])
        boxes = kwargs.get("boxes", [])
        heightfields = kwargs.get("heightfields", [])
        meshes = kwargs.get("meshes", [])
        urdf_files = kwargs.get("urdf_files", [])
        rgba = kwargs.get("rgba", [0.5, 0.5, 0.5, 0.7])

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

        for file_path, mesh_scale, frame_position in heightfields:

            shape = pyb.createCollisionShape(
                shapeType=pyb.GEOM_HEIGHTFIELD,
                meshScale=mesh_scale,
                fileName=str(file_path),
                flags=pyb.GEOM_FORCE_CONCAVE_TRIMESH,
                collisionFramePosition=frame_position,
                physicsClientId=self._id
            )

            shape_id = pyb.createMultiBody(0, shape, physicsClientId=self._id)

            objects.append(shape_id)

        for file_path, mesh_scale, frame_position in meshes:

            shape = pyb.createCollisionShape(
                shapeType=pyb.GEOM_MESH,
                meshScale=mesh_scale,
                fileName=str(file_path),
                flags=pyb.GEOM_FORCE_CONCAVE_TRIMESH,
                collisionFramePosition=frame_position,
                physicsClientId=self._id
            )

            shape_id = pyb.createMultiBody(0, shape, physicsClientId=self._id)

            objects.append(shape_id)

        for file_path, base_position, scale in urdf_files:

            shape_id = pyb.loadURDF(
                fileName=str(file_path),
                basePosition=base_position,
                globalScaling=scale,
                physicsClientId=self._id
            )

            # shape_id = pyb.createMultiBody(0, shape, physicsClientId=self._id)
            objects.append(shape_id)

        for obj_id in objects:
            pyb.changeVisualShape(objectUniqueId=obj_id, linkIndex=-1, textureUniqueId=0, rgbaColor=rgba, physicsClientId=self._id)

        return objects

    def set_config(self, config):
        pos = config[0:3]
        ori = config[3:6]

        q = quaternion_from_euler(*ori)

        pyb.resetBasePositionAndOrientation(
            self._robot, pos, q, physicsClientId=self._id
        )

        if self._track_robot:
            pyb.resetDebugVisualizerCamera(cameraDistance=10,
                                           cameraYaw=-90,
                                           cameraPitch=-20,
                                           cameraTargetPosition=pos,
                                           physicsClientId=self._id)

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

        # start = list(config[0:3])
        # target = [config[0], config[1], 5000]

        # Important! Check the config is not under the terrain map.
        if (pyb.rayTest(list(config[0:3]), [config[0], config[1], 5000], physicsClientId=self._id)[0][0]
                in self._terrain_collision_objects):
            return True

        return self._col_detector.in_collision(margin=self._collision_margin)

    @property
    def key_configs(self):
        return self._key_configs

    @property
    def goal_configs(self):
        return self._goal_configs

    def sample_key_configs(self):

        if np.random.uniform() < 0.5:
            return self.sample_goal_configs()

        lower_bounds = (-40, -30, 5)
        upper_bounds = (-20, 30, 40)
        return tuple(np.random.uniform(low=lower_bounds, high=upper_bounds, size=(1, 3))[0]) + (0, 0, 0)

    def sample_goal_configs(self):
        return tuple(self.goal_configs[np.random.randint(len(self.goal_configs))])


def main():

    env_gui = Corsica(gui=True, debugger=True, dz_level=2, camera_yaw=-350)

    # create user debug parameters
    robot_params_gui = create_robot_params_gui(env_gui._id, start_vals=(0, 0, 0, 0, 0, 0))

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