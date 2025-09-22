import pathlib
import numpy as np

import pomdp_py
import time

from pomdp_py.problems.motion_planning.nav_3d_continuous.nav_3d_continuous import Nav3DContinuousPOMDP, Nav3DState
from pomdp_py.problems.motion_planning.environments.maze_3d import Maze3D
from pomdp_py.problems.motion_planning.test_utils.test import run_experiments

np.set_printoptions(precision=3, suppress=True)  # for neat printing of numpy arrays.

def main(planner_name="porpi", env_name="maze_3d",
         planning_time=1, num_sims=-1,
         runs=50, primitive_steps=500, discount_factor=0.99,
         file_logging=True, prm_logging=True, plot_prm=False,
         visualize_sims=False, visualize_world=True, visualize_belief=False,
         max_macro_action_length=5, start=0):

    # Step 1. Instantiate the motion planning environment.
    candidate_init_positions = [(-28, 15, 0),
                                (-28, -11, 0)]
    if start == 0:
        init_pos = (-28, 15, 0)
    elif start == 1:
        init_pos = (-28, -11, 0)
    else:
        raise ValueError('start must be either 0 for "top" or 1 for "bottom".')

    if env_name.lower() == "maze_3d":
        pyb_env_gui = Maze3D(init_pos + (0, 0, 0), False, False)
        # pyb_env_gui = Maze3D(init_pos + (0, 0, 0), True, False)
        pyb_env = Maze3D(init_pos + (0, 0, 0), False, False)
    else:
        raise ValueError("Unrecognized environment name.")

    # Step 2. Instantiate the POMDP of interest.
    init_state = Nav3DState(init_pos,
                       danger_zone=pyb_env.dz_checker(init_pos+(0,0,0)),
                       landmark=pyb_env.lm_checker(init_pos+(0,0,0)),
                       goal=pyb_env.goal_checker(init_pos+(0,0,0)))

    init_belief = pomdp_py.Particles([Nav3DState(pos,
                       danger_zone=pyb_env.dz_checker(pos+(0,0,0)),
                       landmark=pyb_env.lm_checker(pos+(0,0,0)),
                       goal=pyb_env.goal_checker(pos+(0,0,0))) for pos in candidate_init_positions]*100)

    mp_problem = Nav3DContinuousPOMDP(init_state, init_belief,
                                      pyb_env_gui if visualize_sims else pyb_env, pyb_env_gui,
                                      max_macro_action_length=max_macro_action_length,
                                      prm_nodes=400, prm_assert_connected=True)

    if plot_prm:
        mp_problem.agent.policy_model._prm.plot_prm(mp_problem._pyb_env_gui._id)

    # Step 3. Instantiate the planner.
    if planner_name.lower() == "porpi":
        planner = pomdp_py.PORPI(planning_time=planning_time, num_sims=num_sims,
                                 max_depth=40, rollout_depth=40,
                                 discount_factor=discount_factor, temperature=np.inf,
                                 action_branching_factor=5,
                                 exploitation_prob=0.5,
                                 exploitation_temp=.01,
                                 num_sir_particles=1000, sir_temp=1, sir_zero_weight_exponent=-700,
                                 show_progress=True)
    elif planner_name.lower() == "rg_pomcp":
        # TODO: Clean up parameter naming.
        planner = pomdp_py.RGPOMCP(planning_time=planning_time, num_sims=num_sims,
                                   max_depth=120, rollout_depth=120,
                                   discount_factor=discount_factor, pref_init=0,
                                   exploitation_prob=0.5,
                                   exploration_const=10000,
                                   num_sir_particles=1200, sir_temp=1, sir_zero_weight_exponent=-700,
                                   reward_scale=1.,
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
                                        discount_factor=discount_factor, reward_scaling_factor=1/1000,
                                        num_sir_particles=1000, sir_temp=1, sir_zero_weight_exponent=-700,
                                        show_progress=True)
    elif planner_name.lower() == "refpol":
        planner = pomdp_py.RefPol(discount_factor=discount_factor,
                                  num_sir_particles=1000, sir_temp=1, sir_zero_weight_exponent=-700,
                                  reseed_global_objective=True)
    else:
        planner = pomdp_py.POMCP(planning_time=planning_time, num_sims=num_sims,
                                 max_depth=40, discount_factor=discount_factor, exploration_const=np.sqrt(5),
                                 num_sir_particles=1000, sir_temp=1, sir_zero_weight_exponent=-700,
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
                    visualize_world=visualize_world, visualize_belief=visualize_belief,
                    seeds=list(range(1000, 1051)))

if __name__ == '__main__':
    main(planner_name="pomcp", env_name="maze_3d",
         planning_time=5, num_sims=-1,
         runs=5, primitive_steps=500,
         file_logging=True, plot_prm=False,
         visualize_sims=True, visualize_belief=False,
         max_macro_action_length=10, start=0)