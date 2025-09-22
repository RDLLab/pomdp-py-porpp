from holonomic_rescue import *

def animate(problem, init_state, plot_prm=False, env_evolution_steps=[]):
    """
    Use to test the model plugins based on keyboard inputs.
    Modify as required.
    plot_prm (bool): If True, plots the PRM to the GUI.
    """

    discount_factor = 0.99

    if plot_prm:
        problem.agent.policy_model._prm.plot_prm(pyb_env_gui._id)

    # Step 3. Animate.
    left = pyb.B3G_LEFT_ARROW
    right = pyb.B3G_RIGHT_ARROW
    up = 65297
    down = 65298
    space = 32
    a = 97
    z = 122
    x = 120
    c = 99

    total_steps = 0
    global_objective = None
    env_index = 0

    print("Animating...")
    while True:

        if env_index < len(env_evolution_steps):
            if total_steps >= env_evolution_steps[env_index]:
                env_index += 1
                problem.update_environment(env_index)

        keys = pyb.getKeyboardEvents()

        action = None

        if left in keys and keys[left]:
            action = Nav3DAction(np.pi/2, 0)
        elif right in keys and keys[right]:
            action = Nav3DAction(-np.pi/2, 0)
        elif up in keys and keys[up]:
            action = Nav3DAction(0, 0)
        elif down in keys and keys[down]:
            action = Nav3DAction(np.pi, 0)
        elif a in keys and keys[a]:
            action = Nav3DAction(0, np.pi/2)
        elif z in keys and keys[z]:
            action = Nav3DAction(0, -np.pi/2)
        elif space in keys and keys[space]:
            pyb.resetDebugVisualizerCamera(cameraDistance=20,
                                           cameraYaw=-90,
                                           cameraPitch=-30,
                                           cameraTargetPosition=problem.env.state._position[0:3],
                                           physicsClientId=pyb_env_gui._id)

        if x in keys and keys[x]:

            global_objective = problem.agent.policy_model.maintain_global_objective(problem.env.state, global_objective)
            action = problem.agent.policy_model.sample(problem.env.state, global_objective)

        if action is not None:

            ns, obs, r, nsteps = problem.agent.blackbox_model.sample(state=problem.env.state,
                                                                     action=action,
                                                                     discount_factor=discount_factor)

            total_steps += 1

            problem.env.apply_transition(ns)
            pyb_env.set_config(ns._position + (0, 0, 0))
            problem.visualize_world()

            print(
                f"---\n"
                f"Action Taken: {action}\n"
                f"Observation Received: {obs}\n"
                f"Reward Received: {r}\n"  
                f"Next State: {problem.env.state}\n" 
                f"Global Objective: {global_objective}\n"
                f"total_steps: {total_steps}\n"
                f"nsteps: {nsteps}")

            problem.agent.update_history(action, obs)

            if problem.env.state.terminal:
                print("---\n  Simulation Ended.")
                time.sleep(1.0)
                problem.env.apply_transition(init_state)
                total_steps = 0
                env_index = 0
                global_objective = None

        time.sleep(.1)

if __name__ == '__main__':

    # Step 1. Instantiate the PyBullet environment.
    candidate_init_positions = [(-100, 0, 8)]
    init_pos = (-100, 0, 8)

    pyb_env_gui = Corsica(init_pos+(0,0,0), True, False, dz_level=0)
    pyb_env = Corsica(init_pos+(0,0,0), False, False, dz_level=0)

    # Step 2. Instantiate the POMDP of interest.
    init_state = Nav3DState(init_pos,
                       collision=pyb_env.collision_checker(init_pos+(0,0,0)),
                       no_fly_zone=pyb_env.dz_checker(init_pos+(0,0,0)),
                       landmark=pyb_env.lm_checker(init_pos+(0,0,0)),
                       objectives_reached=[],
                       goal=False)

    init_belief = pomdp_py.Particles([Nav3DState(pos,
                       collision=pyb_env.collision_checker(init_pos+(0,0,0)),
                       no_fly_zone=pyb_env.dz_checker(init_pos+(0,0,0)),
                       landmark=pyb_env.lm_checker(init_pos+(0,0,0)),
                       objectives_reached=[], goal=False) for pos in candidate_init_positions]*1000)

    problem = HolonomicRescuePOMDP(init_state, init_belief, pyb_env_gui, pyb_env_gui,
                               max_macro_action_length=3,
                               prm_nodes=100)

    animate(problem, init_state=init_state, plot_prm=False, env_evolution_steps=[])