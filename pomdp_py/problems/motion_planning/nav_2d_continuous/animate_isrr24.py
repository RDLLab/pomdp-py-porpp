from nav_2d_continuous_isrr24 import *
from pomdp_py.problems.motion_planning.environments.maze_2d_isrr24 import Maze2D

def animate(plot_prm=False, max_macro_action_length=1):
    """
    Use to test the model plugins based on keyboard inputs.
    Modify as required.
    plot_prm (bool): If True, plots the PRM to the GUI.
    """
    # Step 1. Instantiate the PyBullet environment.
    candidate_init_positions = [(21.5, -11.5), (21.5, 10.5)]
    init_pos = (21.5, -11.5) # the true initial state.

    pyb_env_gui = Maze2D(init_pos + (0, 0, 0, 0), True, True)
    pyb_env = Maze2D(init_pos + (0, 0, 0, 0), False, False)

    # Step 2. Instantiate the POMDP of interest.
    init_state = Nav2DState(init_pos,
                       danger_zone=pyb_env.dz_checker(init_pos+(0,0,0,0)),
                       landmark=pyb_env.lm_checker(init_pos+(0,0,0,0)),
                       goal=pyb_env.goal_checker(init_pos+(0,0,0,0)))

    init_belief = pomdp_py.Particles([Nav2DState(pos,
                       danger_zone=pyb_env.dz_checker(init_pos+(0,0,0,0)),
                       landmark=pyb_env.lm_checker(init_pos+(0,0,0,0)),
                       goal=pyb_env.goal_checker(init_pos+(0,0,0,0))) for pos in candidate_init_positions]*1000)

    problem = Nav2DContinuousPOMDP(init_state, init_belief, pyb_env, pyb_env_gui, max_macro_action_length=max_macro_action_length)

    discount_factor = 0.99

    if plot_prm:
        problem.agent.policy_model._prm.plot_prm(pyb_env_gui._id)

    # Step 3. Animate.
    left = pyb.B3G_LEFT_ARROW
    right = pyb.B3G_RIGHT_ARROW
    up = 65297
    down = 65298
    space = 32

    global_objective = None
    total_steps = 0

    print("Animating...")
    while True:

        keys = pyb.getKeyboardEvents()

        if space in keys and keys[space] & pyb.KEY_WAS_TRIGGERED:
            global_objective = problem.agent.policy_model.maintain_global_objective(problem.env.state, global_objective)
            print("---\nCurrent Objective:", global_objective)
            action = problem.agent.policy_model.sample(problem.env.state, global_objective)
        elif left in keys and keys[left] & pyb.KEY_WAS_TRIGGERED:
            action = Nav2DAction(np.pi)
        elif right in keys and keys[right] & pyb.KEY_WAS_TRIGGERED:
            action = Nav2DAction(0)
        elif up in keys and keys[up] & pyb.KEY_WAS_TRIGGERED:
            action = Nav2DAction(np.pi / 2)
        elif down in keys and keys[down] & pyb.KEY_WAS_TRIGGERED:
            action = Nav2DAction(-np.pi / 2)
        else:
            action = None

        if action is not None:
            ns, obs, r, nsteps = pomdp_py.sample_explict_models(T=problem.env.transition_model,
                                                                O=problem.agent.observation_model,
                                                                R=problem.env.reward_model,
                                                                  state=problem.env.state,
                                                                action=action,
                                                                discount_factor=discount_factor)

            problem.env.apply_transition(ns)
            pyb_env.set_config(ns._position + (0, 0, 0, 0))
            problem.visualize_world()

            total_steps += 1

            print(
                f"---\n"
                f"nsteps: {nsteps}\n"
                f"Total Steps: {total_steps}\n"
                f"Action: {action}\n"
                f"Observation: {obs}\n"
                f"Reward: {r}\n"
                f"State: {problem.env.state}")

            problem.agent.update_history(action, obs)


            if problem.env.state.terminal:
                print("Simulation Ended.")
                init_belief = problem.agent.init_belief
                init_state = init_belief.random()
                problem.env.apply_transition(init_state)
                pyb_env.set_config(init_state._position + (0, 0, 0, 0))
                global_objective = None

            time.sleep(1.0/10)

if __name__ == '__main__':
    animate(plot_prm=False, max_macro_action_length=5)