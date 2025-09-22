import pathlib
import numpy as np
import pomdp_py
import time

from pomdp_py.problems.motion_planning.nav_2d_continuous.nav_2d_continuous import Nav2DContinuousPOMDP, Nav2DState
from pomdp_py.problems.motion_planning.environments.maze_2d import Maze2D
from pomdp_py.problems.motion_planning.test_utils.test import run_experiments

np.set_printoptions(precision=3, suppress=True)  # for neat printing of numpy arrays.

def main(planner_name="porpi", env_name="maze_2d",
         planning_time=1, num_sims=-1,
         runs=50, primitive_steps=500, discount_factor=0.99,
         file_logging=True, prm_logging=True, plot_prm=False,
         visualize_sims=False, visualize_world=True, visualize_belief=False,
         max_macro_action_length=5, start=0):

    # Step 1. Instantiate the motion planning environment.
    candidate_init_positions = [
        (1, -40), (1, -41), (1, -42), (2, -40), (2, -41), (2, -42),
        (1, -13), (1, -14), (1, -15), (2, -13), (2, -14), (2, -15)
    ]

    if start == 0:
        init_pos = (1, -13)
    elif start == 1:
        init_pos = (1, -40)
    else:
        raise ValueError('start must be either 0 for "top" or 1 for "bottom".')

    if env_name.lower() == "maze_2d":
        pyb_env_gui = Maze2D(init_pos + (0, 0, 0, 0), 0.0, True)
        pyb_env = Maze2D(init_pos + (0, 0, 0, 0), 0.0, False)
    else:
        raise ValueError("Unrecognized environment name.")

    # Step 2. Instantiate the POMDP of interest.
    init_state = Nav2DState(init_pos,
                            danger_zone=pyb_env.dz_checker(init_pos + (0, 0, 0, 0)),
                            landmark=pyb_env.lm_checker(init_pos + (0, 0, 0, 0)),
                            goal=pyb_env.goal_checker(init_pos + (0, 0, 0, 0)))

    init_belief = pomdp_py.Particles([Nav2DState(pos,
                                                 danger_zone=pyb_env.dz_checker(pos + (0, 0, 0, 0)),
                                                 landmark=pyb_env.lm_checker(pos + (0, 0, 0, 0)),
                                                 goal=pyb_env.goal_checker(pos + (0, 0, 0, 0))) for pos in
                                      candidate_init_positions] * 100)

    mp_problem = Nav2DContinuousPOMDP(init_state, init_belief,
                                      pyb_env_gui if visualize_sims else pyb_env, pyb_env_gui,
                                      max_macro_action_length=max_macro_action_length)

    if plot_prm:
        mp_problem.agent.policy_model._prm.plot_prm(mp_problem._pyb_env_gui._id)

    # Step 3. Instantiate the planner.
    if planner_name.lower() == "porpi":
        planner = pomdp_py.PORPI(planning_time=planning_time, num_sims=num_sims,
                                 max_depth=40, rollout_depth=40,
                                 discount_factor=discount_factor, temperature=2,
                                 action_branching_factor=3,
                                 exploitation_prob=0.5,
                                 exploitation_temp=0.0005,
                                 num_sir_particles=1000, sir_temp=1, sir_zero_weight_exponent=-700,
                                 show_progress=True)
    elif planner_name.lower() == "refsolver":
        planner = pomdp_py.RefSolver(planning_time=planning_time, num_sims=num_sims,
                                     max_depth=40, rollout_depth=40,
                                     discount_factor=discount_factor, temperature=1/1000,
                                     num_sir_particles=1000, sir_temp=1, sir_zero_weight_exponent=-700,
                                     show_progress=True)
    elif planner_name.lower() == "refsolver-nips":
        planner = pomdp_py.RefSolverNIPS(planning_time=planning_time, num_sims=num_sims,
                                        max_depth=40, rollout_depth=40,
                                        discount_factor=discount_factor,
                                        num_sir_particles=1000, sir_temp=1, sir_zero_weight_exponent=-700,
                                        show_progress=True)
    elif planner_name.lower() == "refpol":
        planner = pomdp_py.RefPol(discount_factor=discount_factor,
                                  num_sir_particles=1000, sir_temp=1, sir_zero_weight_exponent=-700,
                                  reseed_global_objective=True)
    else:
        planner = pomdp_py.POMCP(planning_time=planning_time, num_sims=num_sims,
                                 max_depth=30, discount_factor=discount_factor, exploration_const=np.sqrt(5),
                                 num_visits_init=0, value_init=0,
                                 rollout_policy=mp_problem.agent.policy_model,
                                 show_progress=True)
        if not planner_name.lower() == "pomcp":
            print("Unrecognized planner name. Using default planner.")

    # Step 4. Run experiments.
    run_experiments(problem=mp_problem, planner=planner, runs=runs, primitive_steps=primitive_steps,
                    file_logging=file_logging, prm_logging=prm_logging,
                    log_directory=str(pathlib.Path(__file__).parent.parent /
                                      f'experiment_logs/{env_name}_{start}_{planner_name}_'
                                      f'{time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())}'
                                      f'_pt_{planning_time}'),
                    visualize_world=visualize_world, visualize_belief=visualize_belief)

if __name__ == '__main__':
    main(planner_name="porpi", env_name="maze_2d",
         planning_time=10., num_sims=-1,
         runs=5, primitive_steps=500,
         file_logging=True,
         visualize_sims=True, visualize_belief=False,
         max_macro_action_length=10, start=0)