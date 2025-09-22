from pomdp_py.problems.motion_planning.environments.pyb_env import *
from pomdp_py.problems.motion_planning import pyb_utils
from pathlib import Path

import numpy as np
import pybullet as pyb
import time

np.set_printoptions(precision=3, suppress=True)  # for neat printing of numpy arrays.

TIMESTEP = 1.0/600

def tuple_list_to_list(tuple_list):
    return [item for t in tuple_list for item in t]

def list_to_tuple_list(config_list):
    return [tuple(config_list[3*i:3*(i+1)]) for i in range(len(config_list) // 3)]

class DroneCapture(PyBulletEnv):

    PREDATOR_INIT_POSITIONS = [
        (-1, 1., -1.),
        (-1, -1., -1.),
        (1,  1., 1.),
        (1, -1., 1.)
    ]

    PREY_INIT_POSITIONS = [(0., 8., 0.)]
    CAPTURE_RADIUS = 1.5


    def __init__(self,
                 lower_lims=[-12, -12, -1.7],
                 upper_lims=[12, 12, 1.7],
                 predator_init_positions=PREDATOR_INIT_POSITIONS,
                 prey_init_positions=PREY_INIT_POSITIONS,
                 capture_radius=CAPTURE_RADIUS,
                 gui=False,
                 debugger=False,
                 collision_margin=0.01):

        self._lower_lims = lower_lims
        self._upper_lims = upper_lims
        self._predator_init_positions = predator_init_positions
        self._prey_init_positions = prey_init_positions
        self._capture_radius = capture_radius
        self.debugger = debugger
        self._start_std = 0.5
        self._max_steps_no_teleport = 30
        self._cur_steps_no_teleport = self._max_steps_no_teleport
        self._collision_margin = collision_margin

        # Use the GUI for the execution environment and DIRECT for planning environment.
        self._id = pyb.connect(pyb.GUI) if gui else pyb.connect(pyb.DIRECT)

        if not debugger:
            pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 0)

        pyb.resetDebugVisualizerCamera(cameraDistance=20,
                                             cameraYaw=0.,
                                             cameraPitch=-65,
                                             cameraTargetPosition=[0, 0, 0])

        # Load objects.
        self._predators = self.load_predators()
        self._prey = self.load_prey()
        self._collision_objects = self.load_collision_objects()

        #########################################################
        # PREDATOR COLLISION CHECKING
        #########################################################
        self._col_detector = pyb_utils.CollisionDetector(
            self._id,
            [(pred, _) for _ in self._collision_objects for pred in self._predators]
        )

        #########################################################
        # PREY COLLISION CHECKING
        #########################################################
        self._prey_col_detector = pyb_utils.CollisionDetector(
            self._id,
            [(self._prey[0], _) for _ in self._collision_objects]
        )

        #########################################################
        # GOAL CHECKING
        #########################################################

        self._goal_detector = pyb_utils.CollisionDetector(
            self._id,
            [(pred, prey) for prey in self._prey for pred in self._predators]
        )

    def load_predators(self):

        urdfs = [
            [Path(__file__).parent.parent.parent.parent / "data/quadrotor.urdf", self._predator_init_positions[0],
             [0, 0, 0]],
            [Path(__file__).parent.parent.parent.parent / "data/quadrotor.urdf", self._predator_init_positions[1],
             [0, 0, 0]],
            [Path(__file__).parent.parent.parent.parent / "data/quadrotor.urdf", self._predator_init_positions[2],
             [0, 0, 0]],
            [Path(__file__).parent.parent.parent.parent / "data/quadrotor.urdf", self._predator_init_positions[3],
             [0, 0, 0]],
        ]

        return self.load_primitive_objects(urdfs=urdfs, rgba=[0.9, 0, 0.9, 1])

    def load_prey(self):

        spheres = [
            [self.CAPTURE_RADIUS, [0, 0, 0]]
        ]

        return self.load_primitive_objects(spheres=spheres, rgba=[1, 2, 0, .6])

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
            # [[12.5, 12.5, 1], [0, 0, 3.0], [0, 0, 0]],
            [[12.5, 12.5, 1], [0, 0, -3.0], [0, 0, 0]],
            [[1, 14, 4], [13, 0, 1.5], [0, 0, 0]],
            [[1, 14, 4], [-13, 0, 1.5], [0, 0, 0]],
            [[14, 1, 4], [0, 13, 1.5], [0, 0, 0]],
            [[14, 1, 4], [0, -13, 1.5], [0, 0, 0]]
        ]

        return self.load_primitive_objects(spheres=spheres,
                                           capsules=capsules,
                                           boxes=boxes,
                                           heightfields=heightfields,
                                           rgba=[0.8, 0.8, 0.8, 1.])


    def load_primitive_objects(self, **kwargs):

        spheres = kwargs.get("spheres", [])
        capsules = kwargs.get("capsules", [])
        boxes = kwargs.get("boxes", [])
        heightfields = kwargs.get("heightfields", [])
        urdfs = kwargs.get("urdfs", [])

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

        for file_path, center, euler in urdfs:

            objects.append(pyb.loadURDF(
                fileName=str(file_path),
                basePosition=center,
                baseOrientation=pyb.getQuaternionFromEuler(euler),
                globalScaling=.8,
                physicsClientId=self._id))

        for obj_id in objects:
            pyb.changeVisualShape(objectUniqueId=obj_id, linkIndex=-1, rgbaColor=rgba, physicsClientId=self._id)

        return objects

    def sample_config(self, lower_lims=[-12, -12, -1.7], upper_lims=[12, 12, 1.7], num_agents=4):
        """Samples a uniform distribution inside the box. Doesn't check validity.
        Default limits manually specified to ensure configs are sampled inside the bounding box."""

        return list(np.concatenate(np.random.uniform(lower_lims,
                                               upper_lims,
                                               (num_agents, 3))))

    def sample_free_prey_config(self):
        """Sample a uniform distribution inside the box. Returns a free config for prey."""

        prey_config = self.sample_config(num_agents=len(self.PREY_INIT_POSITIONS))
        while self.prey_collision_checker(prey_config):
            prey_config = self.sample_config(num_agents=len(self.PREY_INIT_POSITIONS))
        return prey_config

    def set_config(self, config):

        ori = (0, 0, 0)
        q = quaternion_from_euler(*ori)

        config_list = list_to_tuple_list(config)

        for i, pos in enumerate(config_list):
            pyb.resetBasePositionAndOrientation(
                i, pos, q, physicsClientId=self._id
            )

    def set_prey_config(self, config):

        pos = config[0:3]
        ori = (0, 0, 0)
        q = quaternion_from_euler(*ori)

        pyb.resetBasePositionAndOrientation(
            self._prey[0], pos, q, physicsClientId=self._id
        )

    def reset(self):
        """Resets the environment to its initial state."""

        self.set_config(self._predator_init_positions)
        self.set_prey_config(self._prey_init_positions)

    def goal_checker(self, config):

        self.set_config(config)

        return self._goal_detector.in_collision(margin=self._collision_margin)

    def prey_collision_checker(self, prey_config):
        """Returns True if in collision, False otherwise."""

        self.set_prey_config(prey_config)

        return self._prey_col_detector.in_collision(margin=self._collision_margin)

    def collision_checker(self, config):
        """Returns True if in collision, False otherwise."""

        self.set_config(config)

        return self._col_detector.in_collision(margin=self._collision_margin)

    def captured(self, config, prey_config):
        """A prey is considered captured if at least one predator is within the capture radius of the prey."""
        # predator_configs = [config[3 * i: 3 * (i + 1)] for i, predator_id in enumerate(self.predator_ids)]
        # captured = [np.linalg.norm(np.array(c) - np.array(prey_config)) < self._capture_radius for c in predator_configs]
        # return np.any(captured)

        self.set_config(config)

        return self._goal_detector.in_collision(margin=self._collision_margin)

def create_params_gui(pyb_env, num_predators=4, num_prey=1,
                      lower_lims=[-25, -25, -5], upper_lims=[25, 25, 5],
                      predator_init_pos=DroneCapture.PREDATOR_INIT_POSITIONS,
                      prey_init_pos=DroneCapture.PREY_INIT_POSITIONS,):
    """Create debug params to set the agent positions from the GUI."""

    params = {}
    for i in range(num_predators):
        for j, name in enumerate(["x", "y", "z"]):
            params["predator " + str(i + 1) + name] = pyb.addUserDebugParameter(
                                                            name,
                                                            rangeMin=lower_lims[j],
                                                            rangeMax=upper_lims[j],
                                                            startValue=predator_init_pos[i][j],
                                                            physicsClientId=pyb_env._id)

    for i in range(num_prey):
        for j, name in enumerate(["x", "y", "z"]):
            params["prey " + str(i + 1) + name] = pyb.addUserDebugParameter(
                                                            name,
                                                            rangeMin=lower_lims[j],
                                                            rangeMax=upper_lims[j],
                                                            startValue=prey_init_pos[i][j],
                                                            physicsClientId=pyb_env._id)

    return params

def read_params_gui(params_gui):
    """Read configurations from the GUI."""
    return np.array(
        [
            pyb.readUserDebugParameter(
                param,
            )
            for param in params_gui.values()
        ]
    )


def main():

    env_gui = DroneCapture(gui=True, debugger=True)

    params = create_params_gui(env_gui)

    while True:
        config = read_params_gui(params)
        predators_config = config[0:12]
        env_gui.set_config(predators_config)

        prey_config = config[12:15]
        env_gui.set_prey_config(prey_config)

        if env_gui.captured(predators_config, prey_config):
            print("Captured Prey")
        elif env_gui.prey_collision_checker(prey_config):
            print("Prey Collision")
        elif env_gui.collision_checker(predators_config):
            print("Collision")
        else:
            print("Free")

        time.sleep(.1)

if __name__ == "__main__":
    main()