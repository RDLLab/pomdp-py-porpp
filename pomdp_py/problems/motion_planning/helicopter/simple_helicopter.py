from pomdp_py.framework.basics import *
from pomdp_py.problems.motion_planning.helicopter.joystick import joystick_handler
import pomdp_py
import numpy as np

SCALING_FACTOR = 100

HELICOPTER_MASS = 6000                      # [kg]
GRAVITATIONAL_CONSTANT = 9.8                # [m/s]
MAIN_COLLECTIVE_MAX_THRUST = 130000         # [N]
LONG_CYCL_COEFF = 1000.
LATI_CYCL_COEFF = 1000.
TAIL_COLL_COEFF = 3000.

FUEL_CAPACITY = 1000                            # [L]
MAX_FUEL_CONSUMPTION = 0.05                     # [L/s]
HALF_EXTENTS_XYZ = [15, 5, 2.5]                 # [m]

X_HALF = HALF_EXTENTS_XYZ[0]
Y_HALF = HALF_EXTENTS_XYZ[1]
Z_HALF = HALF_EXTENTS_XYZ[2]

A_x = 5 * 2.5
A_y = 15 * 2.5
A_z = 15 * 5

DRAG_COEFF = 1.05
AIR_DENSITY = 1.225                             # [kg/m^3]

I_xx = 8/3 * X_HALF * (Y_HALF**3 * Z_HALF + Y_HALF * Z_HALF**3) / SCALING_FACTOR
I_yy = 8/3 * Y_HALF * (X_HALF**3 * Z_HALF + X_HALF * Z_HALF**3) / SCALING_FACTOR
I_zz = 8/3 * Z_HALF * (X_HALF**3 * Y_HALF + X_HALF * Y_HALF**3) / SCALING_FACTOR
I_xz = 0.

def euc_dist(array_like1, array_like2):
    return np.linalg.norm(np.array(array_like1) - np.array(array_like2))

class HelicopterState(State):

    XYZ_EPS = 0.1
    UVW_EPS = 0.1
    PQR_EPS = 0.1
    FRAME_EULER_EPS = 0.1
    FUEL_EPS = 0.1
    CONTROL_EPS = 0.1

    def __init__(self, xyz, uvw, pqr, frame_euler, fuel, controls):

        self._xyz = xyz # earth-frame
        self._uvw = uvw # fuselage-frame
        self._pqr = pqr # fuselage-frame
        self._frame_euler = frame_euler # w.r.t earth-frame
        self._fuel = fuel
        self._controls = controls

    @property
    def x(self):
        return self._xyz[0]

    @property
    def y(self):
        return self._xyz[1]

    @property
    def z(self):
        return self._xyz[2]

    @property
    def u(self):
        return self._uvw[0]

    @property
    def v(self):
        return self._uvw[1]

    @property
    def w(self):
        return self._uvw[2]

    @property
    def p(self):
        return self._pqr[0]

    @property
    def q(self):
        return self._pqr[1]

    @property
    def r(self):
        return self._pqr[2]

    @property
    def roll(self):
        return self._frame_euler[0]

    @property
    def pitch(self):
        return self._frame_euler[1]

    @property
    def yaw(self):
        return self._frame_euler[2]

    @property
    def fuel(self):
        return self._fuel

    @property
    def main_col(self):
        return self._controls[0]

    @property
    def tail_col(self):
        return self._controls[1]

    @property
    def long_cyc(self):
        return self._controls[2]

    @property
    def lati_cyc(self):
        return self._controls[3]

    def __hash__(self):
        return 1

    def __eq__(self, other):

        if not isinstance(other, HelicopterState):
            return False
        return (euc_dist(self._xyz, other._xyz) < self.XYZ_EPS and
                euc_dist(self._uvw, other._uvw) < self.UVW_EPS and
                euc_dist(self._pqr, other._pqr) < self.PQR_EPS and
                euc_dist(self._frame_euler, other._frame_euler) < self.FRAME_EULER_EPS and
                euc_dist(self._fuel, other._fuel) < self.FUEL_EPS)

    def __str__(self):
        return (f"State<xyz: {self._xyz} | uvw: {self._uvw} | pqr: {self._pqr} | frame_euler: {self._frame_euler} | fuel: {self._fuel} |\n\tcontrols: {self._controls}>")

    def __repr__(self):
        return self.__str__()

class JoyStickControlAction(Action):

    def __init__(self, controls):

        self._controls = controls

    @property
    def main_col(self):
        return self._controls[0]

    @property
    def tail_col(self):
        return self._controls[1]

    @property
    def long_cyc(self):
        return self._controls[2]

    @property
    def lati_cyc(self):
        return self._controls[3]

    def __hash__(self):
        return hash(self._controls)

    def __eq__(self, other):
        if not isinstance(other, JoyStickControlAction):
            return False
        return self._controls == other._controls

    def __str__(self):
        return (f"Action<MainCol: {round(self.main_col, 2)} | TailCol: {round(self.tail_col, 2)} "
                f"| LongCyc: {round(self.long_cyc, 2)} | LatiCyc: {round(self.lati_cyc, 2)}>")

    def __repr__(self):
        return self.__str__()

class IncrementalControlAction(Action):

    ACTIONS = {'MainCol+', 'MainCol-', 'TailCol+', 'TailCol-', 'LongCyc+', 'LongCyc-', 'LatiCyc+', 'LatiCyc-', 'CenterControls', 'None'}

    def __init__(self, name):

        self._name = name

    @property
    def name(self):
        return self._name

    def __hash__(self):
        return hash(self._name)

    def __eq__(self, other):
        if not isinstance(other, IncrementalControlAction):
            return False
        return self._name == other._name

    def __str__(self):
        return f"Action<{self._name}>"

    def __repr__(self):
        return self.__str__()

class HelicopterObservation(Observation):

    XYZ_EPS = 0.1
    UVW_EPS = 0.1
    PQR_EPS = 0.1
    FRAME_EULER_EPS = 0.1
    FUEL_EPS = 0.1
    CONTROL_EPS = 0.1

    def __init__(self, xyz, uvw, pqr, frame_euler, fuel):

        self._xyz = xyz
        self._uvw = uvw
        self._pqr = pqr
        self._frame_euler = frame_euler
        self._fuel = fuel

    def __hash__(self):
        return 1

    def __eq__(self, other):

        if not isinstance(other, HelicopterObservation):
            return False
        return (euc_dist(self._xyz, other._xyz) < self.XYZ_EPS and
                euc_dist(self._uvw, other._uvw) < self.UVW_EPS and
                euc_dist(self._pqr, other._pqr) < self.PQR_EPS and
                euc_dist(self._frame_euler, other._frame_euler) < self.FRAME_EULER_EPS and
                euc_dist(self._fuel, other._fuel) < self.FUEL_EPS)

    def __str__(self):
        return (f"Observation<xyz: {self._xyz}, uvw: {self._uvw}, pqr: {self._pqr}, frame_euler: {self._frame_euler}, fuel: {self._fuel}")

    def __repr__(self):
        return self.__str__()

class HelicopterTransitionModel(TransitionModel):

    TIME_STEP = 0.05

    STEP_MAIN_COL = 1.
    STEP_TAIL_COL = 2.
    STEP_LONG_CYC = 2.
    STEP_LAT_CYC = 2.

    SD_MAIN_COL = 0.0
    SD_TAIL_COL = 0.0
    SD_LONG_CYC = 0.0
    SD_LAT_CYC = 0.0

    def __init__(self, pyb_env):
        """
        pyb_env: A PyBullet environment.
        """

        self._pyb_env = pyb_env

    def to_earth_frame(self, vector: tuple, frame_euler: tuple):

        r, p, y = frame_euler
        Gamma = np.array([
            [np.cos(p)*np.cos(y), np.cos(p)*np.sin(y), -np.sin(p)],
            [np.sin(r)*np.sin(p)*np.cos(y) - np.cos(r)*np.sin(y), np.sin(r)*np.sin(p)*np.sin(y) + np.cos(r)*np.cos(y), np.sin(r)*np.cos(p)],
            [np.cos(r)*np.sin(p)*np.cos(y) + np.sin(r)*np.sin(y), np.cos(r)*np.sin(p)*np.sin(y) - np.sin(r)*np.cos(y), np.cos(r)*np.cos(p)]
        ])
        return np.matmul(Gamma.transpose(), vector)

    def in_bounds_control(self, control: float, bounds=(-np.pi/2., np.pi/2.)):

        if control > bounds[1]:
            return bounds[1]
        if control < bounds[0]:
            return bounds[0]
        return control

    def simulate_motion(self, state: HelicopterState):

        X = -DRAG_COEFF * AIR_DENSITY * A_x * state.u**2 * np.sign(state.u) / 2 * 80
        Y = -DRAG_COEFF * AIR_DENSITY * A_y * state.v**2 * np.sign(state.v) / 2 * 80
        Z_drag = -DRAG_COEFF * AIR_DENSITY * A_z * state.w**2 * np.sign(state.w) / 2 * 80
        Z = Z_drag - MAIN_COLLECTIVE_MAX_THRUST * state._controls[0]
        L_drag = DRAG_COEFF * AIR_DENSITY * state.p**2 * np.sign(state.p) * np.sqrt(Y_HALF**2 + Z_HALF**2) / .005 # Rough approximation
        M_drag = DRAG_COEFF * AIR_DENSITY * state.q**2 * np.sign(state.q) * np.sqrt(X_HALF**2 + Z_HALF**2) / .005 # Rough approximation
        N_drag = DRAG_COEFF * AIR_DENSITY * state.r**2 * np.sign(state.r) * np.sqrt(X_HALF**2 + Y_HALF**2) / .005 # Rough approximation
        L = LATI_CYCL_COEFF * np.sin(state._controls[3]) - L_drag
        M = -LONG_CYCL_COEFF * np.sin(state._controls[2]) - M_drag
        N = TAIL_COLL_COEFF * state._controls[1] - N_drag

        # Convert to earth-frame coordinates.
        (dx, dy, dz) = self.to_earth_frame(state._uvw, state._frame_euler)

        # The rest are fuselage-frame coordinates.
        du = (state.v * state.r - state.w * state.q) + X / HELICOPTER_MASS + GRAVITATIONAL_CONSTANT * np.sin(state.pitch)
        dv = (state.w * state.p - state.u * state.r) + Y / HELICOPTER_MASS - GRAVITATIONAL_CONSTANT * np.sin(state.roll) * np.cos(state.pitch)
        dw = (state.u * state.q - state.v * state.p) + Z / HELICOPTER_MASS - GRAVITATIONAL_CONSTANT * np.cos(state.roll) * np.cos(state.pitch)
        dp = ((I_yy * I_zz - I_zz**2 - I_xz**2) * state.q * state.r + I_xz * (I_xx - I_yy + I_zz) * state.p * state.q + I_xz * N + I_zz * L)/(I_xx * I_zz - I_xz**2)
        dq = ((I_zz - I_xx) * state.p * state.r + I_xz * (state.r**2 - state.p**2) + M) / I_yy
        dr = (-(I_xx * I_yy - I_xx**2 - I_xz**2) * state.p * state.q - I_xz * (I_xx - I_yy + I_zz) * state.q * state.r + I_xx * N + I_xz * L)/(I_xx * I_zz - I_xz**2)
        d_roll = state.p + state.q * np.sin(state.roll) * np.tan(state.pitch) + state.r * np.cos(state.roll) * np.tan(state.pitch)
        d_pitch = state.q * np.cos(state.roll) - state.r * np.sin(state.roll)
        d_yaw = state.q * np.sin(state.roll) / np.cos(state.pitch) + state.r * np.cos(state.roll) / np.cos(state.pitch)

        # print("XYZ_drag:", X, Y, Z_drag)
        # print("LMN_drag:", L_drag, M_drag, N_drag)
        # print(f"XYZ: {X}, {Y}, {Z}")
        # print(f"LMN: {L}, {M}, {N}")
        # print(f"uvw: {state._uvw}")
        # print("d_uvw:", du, dv, dw)
        # print("d_pqr", dp, dq, dr)
        # print("d_xyz:", dx, dy, dz)
        # print(f"d_rpy: {d_roll}, {d_pitch}, {d_yaw}")

        xyz = (state.x + dx * self.TIME_STEP, state.y + dy * self.TIME_STEP, state.z + dz * self.TIME_STEP)
        uvw = (state.u + du * self.TIME_STEP, state.v + dv * self.TIME_STEP, state.w + dw * self.TIME_STEP)
        pqr = (state.p + dp * self.TIME_STEP, state.q + dq * self.TIME_STEP, state.r + dr * self.TIME_STEP)
        frame_euler = (state.roll + d_roll * self.TIME_STEP, state.pitch + d_pitch * self.TIME_STEP, state.yaw + d_yaw * self.TIME_STEP)

        return xyz, uvw, pqr, frame_euler

    def simulate_control(self, state: HelicopterState, action: IncrementalControlAction):

        if action.name == 'MainCol+':
            main_col_step = self.STEP_MAIN_COL + np.random.normal(0., self.SD_MAIN_COL)
        elif action.name == 'MainCol-':
            main_col_step = -self.STEP_MAIN_COL + np.random.normal(0., self.SD_MAIN_COL)
        else:
            main_col_step = 0.

        if action.name == 'TailCol+':
            tail_col_step = self.STEP_TAIL_COL + np.random.normal(0., self.SD_TAIL_COL)
        elif action.name == 'TailCol-':
            tail_col_step = -self.STEP_TAIL_COL + np.random.normal(0., self.SD_TAIL_COL)
        else:
            tail_col_step = 0.

        if action.name == 'LongCyc+':
            long_cyc_step = +self.STEP_LONG_CYC + np.random.normal(0., self.SD_LONG_CYC)
        elif action.name == 'LongCyc-':
            long_cyc_step = -self.STEP_LONG_CYC + np.random.normal(0., self.SD_LONG_CYC)
        else:
            long_cyc_step = 0.

        if action.name == 'LatiCyc+':
            lati_cyc_step = +self.STEP_LAT_CYC + np.random.normal(0., self.SD_LAT_CYC)
        elif action.name == 'LatiCyc-':
            lati_cyc_step = -self.STEP_LAT_CYC + np.random.normal(0., self.SD_LAT_CYC)
        else:
            lati_cyc_step = 0.

        if action.name == 'CenterControls':
            return state.main_col, 0., 0., 0.

        main_col = self.in_bounds_control(state.main_col + self.TIME_STEP * main_col_step, bounds=(0., 1.))
        tail_col = self.in_bounds_control(state.tail_col + self.TIME_STEP * tail_col_step)
        long_cyc = self.in_bounds_control(state.long_cyc + self.TIME_STEP * long_cyc_step)
        lati_cyc = self.in_bounds_control(state.lati_cyc + self.TIME_STEP * lati_cyc_step)

        return main_col, tail_col, long_cyc, lati_cyc

    def sample(self, state: HelicopterState, action: Action):

        # Simulate transition effect on state and controls
        xyz, uvw, pqr, frame_euler = self.simulate_motion(state)

        if isinstance(action, IncrementalControlAction):
            controls = self.simulate_control(state, action)
        elif isinstance(action, JoyStickControlAction):
            controls = action._controls
        else:
            raise(ValueError, "Unrecognized control action.")

        fuel = state.fuel - self.TIME_STEP * MAX_FUEL_CONSUMPTION * state.main_col

        return HelicopterState(xyz, uvw, pqr, frame_euler, fuel, controls)

class HelicopterObservationModel(ObservationModel):

    def __init__(self, pyb_env):
        """
        pyb_env: A PyBullet environment.
        """

        self._pyb_env = pyb_env

    def sample(self, next_state: HelicopterState, action: Action):

        return HelicopterObservation(next_state._xyz,
                                     next_state._uvw,
                                     next_state._pqr,
                                     next_state._frame_euler,
                                     next_state._fuel)

class HelicopterRewardModel(RewardModel):

    def __init__(self, pyb_env):
        """
        pyb_env: A PyBullet environment.
        """

        self._pyb_env = pyb_env

    def sample(self, state: HelicopterState, action: Action, next_state: HelicopterState):

        return 0

class SimpleHelicopterPOMDP(MPPOMDP):

    def __init__(self,
                 init_state,
                 init_belief,
                 pyb_env,
                 pyb_env_gui):

        self._init_state = init_state
        self._init_belief = init_belief
        self._pyb_env = pyb_env
        self._pyb_env_gui = pyb_env_gui

        "Agent"
        agent = pomdp_py.Agent(init_belief=init_belief,
                               policy_model=None,
                               transition_model=HelicopterTransitionModel(pyb_env=pyb_env),
                               observation_model=HelicopterObservationModel(pyb_env=pyb_env),
                               reward_model=HelicopterRewardModel(pyb_env=pyb_env),
                               name="Helicopter")

        "Environment"
        env = pomdp_py.Environment(init_state=init_state,
                                   transition_model=HelicopterTransitionModel(pyb_env=pyb_env_gui),
                                   reward_model=HelicopterRewardModel(pyb_env=pyb_env_gui))

        super().__init__(agent, env, name="SimpleHelicopterPOMDP")

    def visualize_world(self):
        rpy = self.env.state._frame_euler
        ypr_sign_corrected = (rpy[2], -rpy[1], -rpy[0]) # PyBullet uses clockwise convention for roll and pitch (not yaw).
        self._pyb_env_gui.set_config(self.env.state._xyz + ypr_sign_corrected)





