# TODO: This environment is a work-in-progress.

from pomdp_py.problems.motion_planning.environments.pyb_env import *

import numpy as np
import pybullet as pyb
import time
from pathlib import Path
import fcl

np.set_printoptions(precision=3, suppress=True)  # for neat printing of numpy arrays.

TIMESTEP = 1.0/600
class FCLMaze3D(PyBulletEnv):

    def __init__(self,
                 robot_init_config=(.0, .0, .0, .0, .0, .0),
                 gui=False,
                 debugger=False):

        # Use the GUI for the execution environment and DIRECT for planning environment.
        self._id = pyb.connect(pyb.GUI) if gui else pyb.connect(pyb.DIRECT)

        if not debugger:
            pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 0)

        pyb.resetDebugVisualizerCamera(cameraDistance=50,
                                       cameraYaw=0,
                                       cameraPitch=-80,
                                       cameraTargetPosition=[0, -0, 0],
                                       physicsClientId=self._id)

        self._robot_init_config = robot_init_config

        # Load objects.
        self._robot, self._robot_vis = self.load_robot()
        self._goals, self._goals_vis = self.load_goals()
        self._landmarks, self._landmarks_vis = self.load_landmarks()
        self._danger_zones, self._danger_zones_vis = self.load_danger_zones()
        self._collision_objects, self._collision_objects_vis = self.load_collision_objects()

        # Set up collision managers.
        self._goal_manager = fcl.DynamicAABBTreeCollisionManager()
        self._goal_manager.registerObjects(self._goals)
        self._goal_manager.setup()

        self._landmark_manager = fcl.DynamicAABBTreeCollisionManager()
        self._landmark_manager.registerObjects(self._landmarks)
        self._landmark_manager.setup()

        self._dz_manager = fcl.DynamicAABBTreeCollisionManager()
        self._dz_manager.registerObjects(self._danger_zones)
        self._dz_manager.setup()

        self._collision_obj_manager = fcl.DynamicAABBTreeCollisionManager()
        self._collision_obj_manager.registerObjects(self._collision_objects)
        self._collision_obj_manager.setup()

    def load_robot(self):

        capsule = [
            [0.6, 2.0, self._robot_init_config[0:3], self._robot_init_config[3:6]]
        ]

        return self.load_primitive_objects(capsules=capsule)[0], self.visualize_primitive_objects(capsules=capsule, rgba=[0, 1, 1, 1])[0]

    def load_goals(self, visualize=True):

        spheres = [
            [1.5, [17, -7, 6]],
            [1.5, [27, 10, -4]]
        ]

        return self.load_primitive_objects(spheres=spheres), self.visualize_primitive_objects(spheres=spheres, rgba=[0, 1, 0, 1])

    def load_landmarks(self):

        boxes = [
            [[2, 2, 2], [-26, -20, -5], [0, 0, 0]],
            [[2, 2, 2], [-20, -25, -5], [0, 0, 0]],
            [[2, 2, 2], [2, -9, -5], [0, 0, 0]],
            [[2, 2, 2], [7, 8, 0], [0, 0, 0]],
            [[3, 3, 3], [17, -7, 6], [0, 0, 0]],
            [[3, 3, 3], [27, 10, -4], [0, 0, 0]]
        ]

        return self.load_primitive_objects(boxes=boxes), self.visualize_primitive_objects(boxes=boxes, rgba=[1, 0.1, 1, 0.5])

    def load_danger_zones(self):

        boxes = [
            [[10, 1, 6], [-11, 17, 2.0], [0, 0, 0]],
            [[10, 1, 4], [-11, 13, 2.0], [0, 0, 0]],
            [[5, 1, 4], [-5, -26, 2.0], [0, 0, 0]]
        ]

        return self.load_primitive_objects(boxes=boxes), self.visualize_primitive_objects(boxes=boxes, rgba=[1, 1, 0.2, 0.8])

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
            [[30, 30, 2], [0.0, 0.0, -10.0], [0, 0, 0]],
            [[2, 32, 10], [32, 0, 0], [0, 0, 0]],
            [[2, 32, 10], [-32, 0, 0], [0, 0, 0]],
            [[32, 2, 10], [0, 32, 0], [0, 0, 0]],
            [[32, 2, 10], [0, -32, 0], [0, 0, 0]],
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

        return (self.load_primitive_objects(spheres=spheres,
                                            capsules=capsules,
                                            boxes=boxes,
                                            heightfields=heightfields),
                self.visualize_primitive_objects(spheres=spheres,
                                            capsules=capsules,
                                            boxes=boxes,
                                            heightfields=heightfields))


    def load_primitive_objects(self, **kwargs):

        spheres = kwargs.get("spheres", [])
        capsules = kwargs.get("capsules", [])
        boxes = kwargs.get("boxes", [])
        heightfields = kwargs.get("heightfields", [])

        objects = []

        for radius, center in spheres:
            sphere = fcl.Sphere(radius)
            t = fcl.Transform(np.array(center))
            obj = fcl.CollisionObject(sphere, t)
            objects.append(obj)


        for radius, height, center, euler in capsules:
            capsule = fcl.Capsule(radius, height)
            q = quaternion_from_euler(*euler)
            t = fcl.Transform(q, np.array(center))
            obj = fcl.CollisionObject(capsule, t)
            objects.append(obj)


        for half_extents, center, euler in boxes:
            lengths = half_extents_to_lengths(*half_extents)
            box = fcl.Box(*lengths)
            q = quaternion_from_euler(*euler)
            t = fcl.Transform(q, np.array(center))
            obj = fcl.CollisionObject(box, t)
            objects.append(obj)

        return objects

    def visualize_primitive_objects(self, **kwargs):
        spheres = kwargs.get("spheres", [])
        capsules = kwargs.get("capsules", [])
        boxes = kwargs.get("boxes", [])
        heightfields = kwargs.get("heightfields", [])
        rgba = kwargs.get("rgba", [1, 1, 1, 0.9])

        objects = []

        for radius, center in spheres:
            objects.append(pyb.createMultiBody(0,
                                             pyb.createCollisionShape(pyb.GEOM_SPHERE,
                                                                       radius=radius),
                                             basePosition=center,
                                             physicsClientId=self._id))

        for radius, height, center, euler in capsules:
            objects.append(pyb.createMultiBody(0,
                                             pyb.createCollisionShape(pyb.GEOM_CAPSULE,
                                                                       radius=radius,
                                                                       height=height),
                                             basePosition=center,
                                             baseOrientation=pyb.getQuaternionFromEuler(euler),
                                             physicsClientId=self._id))

        for half_extents, center, euler in boxes:
            objects.append(pyb.createMultiBody(0,
                                               pyb.createCollisionShape(pyb.GEOM_BOX,
                                                                        halfExtents=half_extents),
                                               basePosition=center,
                                               baseOrientation=pyb.getQuaternionFromEuler(euler),
                                               physicsClientId=self._id))

        for file_path, mesh_scale in heightfields:

            shape = pyb.createCollisionShape(
                shapeType=pyb.GEOM_HEIGHTFIELD,
                meshScale=mesh_scale,
                fileName=str(file_path)
            )

            shape_id = pyb.createMultiBody(0, shape, physicsClientId=self._id)
            objects.append(shape_id)

        for obj_id in objects:
            pyb.changeVisualShape(objectUniqueId=obj_id, linkIndex=-1, rgbaColor=rgba)

        return objects

    def set_config(self, config):
        pos = config[0:3]
        ori = config[3:6]

        q = quaternion_from_euler(*ori)

        self._robot.setTranslation(np.array(pos))
        self._robot.setQuatRotation(np.array(q))

        pyb.resetBasePositionAndOrientation(
            self._robot_vis, pos, q, physicsClientId=self._id
        )

    def reset(self):
        """Resets the environment to its initial state."""

        self.set_config(self._robot_init_config)

    def goal_checker(self, config):

        self.set_config(config)

        goal_data = fcl.CollisionData(request=fcl.CollisionRequest(num_max_contacts=1, enable_contact=True))
        self._goal_manager.collide(self._robot, goal_data, fcl.defaultCollisionCallback)
        return goal_data.result.is_collision

    def lm_checker(self, config):

        self.set_config(config)

        lm_data = fcl.CollisionData(request=fcl.CollisionRequest(num_max_contacts=1, enable_contact=True))
        self._landmark_manager.collide(self._robot, lm_data, fcl.defaultCollisionCallback)
        return lm_data.result.is_collision

    def dz_checker(self, config):

        self.set_config(config)

        dz_data = fcl.CollisionData(request=fcl.CollisionRequest(num_max_contacts=1, enable_contact=True))
        self._dz_manager.collide(self._robot, dz_data, fcl.defaultCollisionCallback)
        return dz_data.result.is_collision

    def collision_checker(self, config):

        self.set_config(config)

        collision_data = fcl.CollisionData(request=fcl.CollisionRequest(num_max_contacts=1, enable_contact=True))
        self._collision_obj_manager.collide(self._robot, collision_data, fcl.defaultCollisionCallback)
        return collision_data.result.is_collision

    def sample_key_configs(self):

        milestones = [
            (-26, -20, -4),
            (-20, -25, -4),
            (2, -9, -4),
            (7, 8, 0),
            (17, -7, 3),
            (27, 10, -2)
        ]

        return tuple(milestones[np.random.randint(len(milestones))]) + (0, 0, 0)

    def sample_goal_configs(self):

        goals = [
            (17, -7, 3),
            # (27, 10, -2)
        ]

        return tuple(goals[np.random.randint(len(goals))]) + (0, 0, 0)

def main():

    env_gui = FCLMaze3D(gui=True, debugger=True)

    # create user debug parameters
    robot_params_gui = create_robot_params_gui(env_gui._id)

    while True:
        config = read_robot_params_gui(robot_params_gui, client_id=env_gui._id)
        # env_gui.set_config(config)

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