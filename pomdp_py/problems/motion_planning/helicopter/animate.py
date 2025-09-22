from pomdp_py.problems.motion_planning.pyb_utils import quatx
from simple_helicopter import *
from pomdp_py.problems.motion_planning.environments.corsica import Corsica
from pomdp_py.problems.motion_planning.helicopter.joystick import *
import pybullet as pyb
import pomdp_py
import time
import pygame
from pygame.locals import *

def animate():

    init_state = HelicopterState(xyz=(-200,0,2),
                                 uvw=(0,0,0),
                                 pqr=(0,0,0),
                                 frame_euler=(np.pi,0,0),
                                 fuel=FUEL_CAPACITY,
                                 controls=(.4523,0,0,0))

    init_belief = pomdp_py.Particles([init_state])

    pyb_env_gui = Corsica(gui=True, debugger=False, camera_target=(-100,0,1), track_robot=False)

    problem = SimpleHelicopterPOMDP(init_state=init_state,
                                    init_belief=init_belief,
                                    pyb_env=pyb_env_gui,
                                    pyb_env_gui=pyb_env_gui)

    left = pyb.B3G_LEFT_ARROW
    right = pyb.B3G_RIGHT_ARROW
    up = pyb.B3G_UP_ARROW
    down = pyb.B3G_DOWN_ARROW
    space = 32
    a = 97
    z = 122
    x = 120
    c = 99

    pygame.init()

    # print("Debugging...")
    # while True:

    print("Animating...")
    while True:

        joystick_count = pygame.joystick.get_count()
        joysticks = []
        for i in range(joystick_count):
            joysticks.append(joystick_handler(i))

        action = IncrementalControlAction("None")
        for event in [pygame.event.wait(50), ] + pygame.event.get():
            if event.type == JOYAXISMOTION:
                joysticks[event.joy].axis[event.axis] = event.value
                joy = joysticks[0].axis
                controls = ((1. - joy[3]) * .5, joy[2]*np.pi/2, -joy[1]*np.pi/2, joy[0]*np.pi/2)
                action = JoyStickControlAction(controls)

        keys = pyb.getKeyboardEvents()

        if a in keys and keys[a] & pyb.KEY_WAS_TRIGGERED:
            action = IncrementalControlAction('MainCol+')
            print("----\n", action, "\n----")
        elif z in keys and keys[z] & pyb.KEY_WAS_TRIGGERED:
            action = IncrementalControlAction('MainCol-')
            print("----\n", action, "\n----")
        elif x in keys and keys[x] & pyb.KEY_WAS_TRIGGERED:
            action = IncrementalControlAction('TailCol+')
            print("----\n", action, "\n----")
        elif c in keys and keys[c] & pyb.KEY_WAS_TRIGGERED:
            action = IncrementalControlAction('TailCol-')
            print("----\n", action, "\n----")
        elif left in keys and keys[left] & pyb.KEY_WAS_TRIGGERED:
            action = IncrementalControlAction('LatiCyc-')
            print("----\n", action, "\n----")
        elif right in keys and keys[right] & pyb.KEY_WAS_TRIGGERED:
            action = IncrementalControlAction('LatiCyc+')
            print("----\n", action, "\n----")
        elif up in keys and keys[up] & pyb.KEY_WAS_TRIGGERED:
            action = IncrementalControlAction('LongCyc+')
            print("----\n", action, "\n----")
        elif down in keys and keys[down] & pyb.KEY_WAS_TRIGGERED:
            action = IncrementalControlAction('LongCyc-')
            print("----\n", action, "\n----")
        elif space in keys and keys[space] & pyb.KEY_WAS_TRIGGERED:
            action = IncrementalControlAction('CenterControls')
            print("----\n", action, "\n----")

        pos = problem.env.state._xyz
        pitch = problem.env.state.pitch
        yaw = problem.env.state.yaw
        pyb.resetDebugVisualizerCamera(cameraDistance=15,
                                       cameraYaw=yaw*180/np.pi-90,
                                       cameraPitch=-pitch*180/np.pi-20,
                                       cameraTargetPosition=pos,
                                       physicsClientId=pyb_env_gui._id)

        ns, obs, r, nsteps = pomdp_py.sample_explict_models(T=problem.env.transition_model,
                                                            O=problem.agent.observation_model,
                                                            R=problem.env.reward_model,
                                                            state=problem.env.state,
                                                            action=action,
                                                            discount_factor=.99)

        problem.env.apply_transition(ns)
        print(obs)
        problem.visualize_world()
        time.sleep(0.005)

if __name__ == '__main__':
    animate()
