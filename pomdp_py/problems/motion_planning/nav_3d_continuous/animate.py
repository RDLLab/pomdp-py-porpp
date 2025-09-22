from nav_3d_continuous import *
from pomdp_py.problems.motion_planning.environments.maze_3d import Maze3D

def animate(plot_prm=False, max_macro_action_length=1):
    """
    Use to test the model plugins based on keyboard inputs.
    Modify as required.
    plot_prm (bool): If True, plots the PRM to the GUI.
    """
    # Step 1. Instantiate the PyBullet environment.
    candidate_init_positions = [(-27, 15, 0),
                                (-27, -11, 0)]
    init_pos = (-28, 15, 0) # the true initial state.

    pyb_env_gui = Maze3D(init_pos + (np.pi, 0, 0), True, False)
    pyb_env = Maze3D(init_pos + (np.pi, 0, 0), False, False)

    # Step 2. Instantiate the POMDP of interest.
    init_state = Nav3DState(init_pos,
                       danger_zone=pyb_env.dz_checker(init_pos+(0,0,0)),
                       landmark=pyb_env.lm_checker(init_pos+(0,0,0)),
                       goal=pyb_env.goal_checker(init_pos+(0,0,0)))

    init_belief = pomdp_py.Particles([Nav3DState(pos,
                       danger_zone=pyb_env.dz_checker(init_pos+(0,0,0)),
                       landmark=pyb_env.lm_checker(init_pos+(0,0,0)),
                       goal=pyb_env.goal_checker(init_pos+(0,0,0))) for pos in candidate_init_positions]*1000)

    problem = Nav3DContinuousPOMDP(init_state, init_belief, pyb_env, pyb_env_gui,
                                   max_macro_action_length=max_macro_action_length,
                                   prm_nodes=300)

    # problem.agent.policy_model._prm.plot_prm(problem._pyb_env_gui._id)


    # path = problem.agent.policy_model._prm.shortest_path_offline((17, -7, 6, 0, 0, 0), (17, -7, 6, 0, 0, 0))
    # print(path)

    discount_factor = 0.99

    out_edge = problem.agent.policy_model._prm.sample_out_edge((0, 0, 0, 0, 0, 0))

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

    depth = 0
    global_objective = None

    print("Animating...")
    while True:

        keys = pyb.getKeyboardEvents()

        action = None

        if space in keys and keys[space] & pyb.KEY_WAS_TRIGGERED:
            global_objective = problem.agent.policy_model.maintain_global_objective(problem.env.state, global_objective)
            print(f"---\nCurrent Objective: {global_objective}")
            action = problem.agent.policy_model.sample(problem.env.state, global_objective)
        elif left in keys and keys[left] & pyb.KEY_WAS_TRIGGERED:
            action = Nav3DAction(np.pi, 0)
        elif right in keys and keys[right] & pyb.KEY_WAS_TRIGGERED:
            action = Nav3DAction(0, 0)
        elif up in keys and keys[up] & pyb.KEY_WAS_TRIGGERED:
            action = Nav3DAction(np.pi/2, 0)
        elif down in keys and keys[down] & pyb.KEY_WAS_TRIGGERED:
            action = Nav3DAction(-np.pi/2, 0)
        elif a in keys and keys[a] & pyb.KEY_WAS_TRIGGERED:
            action = Nav3DAction(0, np.pi/2)
        elif z in keys and keys[z] & pyb.KEY_WAS_TRIGGERED:
            action = Nav3DAction(0, -np.pi/2)

        if action is not None:
            ns, obs, r, nsteps = pomdp_py.sample_explict_models(T=problem.env.transition_model,
                                                                O=problem.agent.observation_model,
                                                                R=problem.env.reward_model,
                                                                state=problem.env.state,
                                                                action=action,
                                                                discount_factor=discount_factor)

            problem.env.apply_transition(ns)
            pyb_env.set_config(ns._position + (0, 0, 0))
            problem.visualize_world()

            # path = problem.agent.policy_model._prm.shortest_path_offline((0, 0, 0, 0, 0, 0), curr_target)
            # edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]

            depth += nsteps

            print(
                f"---\n"
                f"Action Taken: {action}\n"
                f"Observation Received: {obs}\n"
                f"Reward Received: {r}\n"
                f"New State: {problem.env.state}\n"
                f"total_steps: {depth}")

            problem.agent.update_history(action, obs)

            if problem.env.state.terminal:
                print("---\n  Simulation Ended.")
                time.sleep(1.0)
                problem.env.apply_transition(init_state)
                pyb_env.set_config(init_state._position + (0, 0, 0))
                global_objective = None

        time.sleep(1.0/10)

if __name__ == '__main__':

    animate(plot_prm=False, max_macro_action_length=10)