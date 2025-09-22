import numpy as np
import pybullet as pyb

np.set_printoptions(precision=3, suppress=True)  # for neat printing of numpy arrays.

TIMESTEP = 1.0/60

def quaternion_from_euler(r, p, y):

    rs2 = np.sin(r / 2)
    rc2 = np.cos(r / 2)
    ps2 = np.sin(p / 2)
    pc2 = np.cos(p / 2)
    ys2 = np.sin(y / 2)
    yc2 = np.cos(y / 2)

    qx = rs2 * pc2 * yc2 - rc2 * ps2 * ys2
    qy = rc2 * ps2 * yc2 + rs2 * pc2 * ys2
    qz = rc2 * pc2 * ys2 - rs2 * ps2 * yc2
    qw = rc2 * pc2 * yc2 + rs2 * ps2 * ys2

    return [qw, qx, qy, qz]

def half_extents_to_lengths(lx, ly, lz):
    return [lx*2, ly*2, lz*2]

cdef class PyBulletEnv():
    """An abstract wrapper class for PyBullet environments."""

    def load_environment(self, client_id):
        """
        Connects to a PyBullet client and sets up robot, obstacles and goals from input (e.g. URDF) files.
        """
        raise NotImplementedError

    def get_goal_pos_ori(self, goal_id):
        """Returns a tuple of (x, y, z, roll, pitch, yaw) for the goal of interest."""

        pos = pyb.getBasePositionAndOrientation(goal_id)[0]
        ori = pyb.getEulerFromQuaternion(pyb.getBasePositionAndOrientation(goal_id)[1])

        return pos + ori

    def set_config(self, config):
        """
        Set the configuration of the robot in the PyBullet environment.
        """
        raise NotImplementedError

    def in_bounds(self, config):
        """
        Takes a configuration and returns True if any part of the robot is in bounds.
        """
        raise NotImplementedError

    def collision_checker(self, config):
        """
        Takes a configuration and returns True if any part of the robot is in collision with an obstacle.
        """
        raise NotImplementedError

    def goal_checker(self, config):
        """
        Takes a configuration and returns True if any part of the robot is inside the goal region.
        """
        raise NotImplementedError

    def ceiling_checker(self, config):
        """
        Takes a configuration and returns True if any part of the robot is below the cloud ceiling.
        """
        raise NotImplementedError

    def dz_checker(self, config):
        """
        Takes a configuration and returns True if any part of the robot is at a danger zone.
        """
        raise NotImplementedError

    def lm_checker(self, config):
        """
        Takes a configuration and returns True if any part of the robot is at a landmark.
        """
        raise NotImplementedError

    def lsr_checker(self, config):
        """
        Takes a configuration and returns True if any port of the robot is at a low stress region.
        """
        raise NotImplementedError

    def reset(self):
        """Resets the environment to its initial state."""

        raise NotImplementedError

    def sample_key_config(self):
        """Samples key user-defined configurations."""

        raise NotImplementedError

    def sample_goal_config(self):
        """Samples goal configurations."""

        raise NotImplementedError

    @property
    def goal_positions(self):
        """Returns list of all goal positions."""

        raise NotImplementedError


def create_robot_params_gui(client_id,
                            names=("x", "y", "z", "yaw", "pitch", "roll"),
                            lower_lims=(-100, -100, -100, -np.pi, -np.pi, -np.pi),
                            upper_lims=(100, 100, 100, np.pi, np.pi, np.pi),
                            start_vals=(0, 0, 0, 0, 0, 0)):

    """Create debug params to set the robot positions from the GUI."""

    params = {}

    for name, lower_lim, upper_lim, start_val in zip(names, lower_lims, upper_lims, start_vals):
        params[name] = pyb.addUserDebugParameter(
            name,
            rangeMin=lower_lim,
            rangeMax=upper_lim,
            startValue=start_val,
            physicsClientId=client_id
        )

    return params

def read_robot_params_gui(robot_params_gui, client_id):

    """Read robot configuration from the GUI."""

    return np.array(
        [
            pyb.readUserDebugParameter(
                param,
                physicsClientId=client_id,
            )
            for param in robot_params_gui.values()
        ]
    )