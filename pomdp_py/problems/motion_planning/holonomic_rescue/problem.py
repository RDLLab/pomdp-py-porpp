from pomdp_py.problems.motion_planning.holonomic_rescue.holonomic_rescue import *
from pomdp_py.problems.motion_planning.test_utils.test import run_experiments

np.set_printoptions(precision=3, suppress=True)  # for neat printing of numpy arrays.

def main(planner_name="porpi", env_name="corsica",
         planning_time=1, num_sims=-1,
         runs=50, primitive_steps=500, discount_factor=0.99,
         file_logging=True, prm_logging=True, plot_prm=False,
         visualize_sims=False, visualize_world=True, visualize_belief=False,
         max_macro_action_length=5,
         env_evolution_steps=[15, 50, 100]):

    # Step 1. Instantiate the motion planning environment.
    candidate_init_positions = [(-100, 0, 15)]
    init_pos = (-100, 0, 15)

    if env_name.lower() == "corsica":
        from pomdp_py.problems.motion_planning.environments.corsica import Corsica
        pyb_env = Corsica(gui=False, debugger=False, dz_level=0)
        # pyb_env_gui = Corsica(gui=True, debugger=False, dz_level=0)
        pyb_env_gui = Corsica(gui=False, debugger=False, dz_level=0)
    elif env_name.lower() == "sydney":
        from pomdp_py.problems.motion_planning.environments.sydney import Sydney
        pyb_env = Sydney(gui=False, debugger=False)
        pyb_env_gui = Sydney(gui=True, debugger=False)
    elif env_name.lower() == "terrain":
        from pomdp_py.problems.motion_planning.environments.terrain import Terrain
        pyb_env = Terrain(gui=False, debugger=False)
        pyb_env_gui = Terrain(gui=True, debugger=False)
    else:
        raise ValueError("Unrecognized environment name.")

    # Step 2. Instantiate the POMDP of interest.
    init_state = Nav3DState(init_pos,
                       collision=pyb_env.collision_checker(init_pos+(0,0,0)),
                       no_fly_zone=pyb_env.dz_checker(init_pos+(0,0,0)),
                       landmark=pyb_env.lm_checker(init_pos+(0,0,0)),
                       objectives_reached=[],
                       goal=pyb_env.goal_checker(init_pos+(0,0,0)))

    init_belief = pomdp_py.Particles([Nav3DState(pos,
                       collision=pyb_env.collision_checker(init_pos + (0, 0, 0)),
                       no_fly_zone=pyb_env.dz_checker(pos+(0,0,0)),
                       landmark=pyb_env.lm_checker(pos+(0,0,0)),
                       objectives_reached=[],
                       goal=pyb_env.goal_checker(pos+(0,0,0))) for pos in candidate_init_positions]*500)

    mp_problem = HolonomicRescuePOMDP(init_state, init_belief,
                                  pyb_env_gui if visualize_sims else pyb_env, pyb_env_gui,
                                  max_macro_action_length=max_macro_action_length,
                                  prm_nodes=500,
                                  prm_lower_bounds=(-100, -100, 0),
                                  prm_upper_bounds=(100, 100, 50))

    if plot_prm:
        mp_problem.agent.policy_model._prm.plot_prm(mp_problem._pyb_env_gui._id)

    # Step 3. Instantiate the planner.
    if planner_name.lower() == "porpi":
        planner = pomdp_py.PORPI(planning_time=planning_time, num_sims=num_sims,
                                 max_depth=50, rollout_depth=50,
                                 discount_factor=discount_factor, temperature=5,
                                 action_branching_factor=5, exploitation_temp=.01,
                                 exploration_const=100,
                                 num_sir_particles=500, sir_temp=1, sir_zero_weight_exponent=-700,
                                 reward_scale=1,
                                 show_progress=True)
    elif planner_name.lower() == "refsolver":
        planner = pomdp_py.RefSolver(planning_time=planning_time, num_sims=num_sims,
                                     max_depth=100, rollout_depth=100,
                                     discount_factor=discount_factor, temperature=.01,
                                     num_sir_particles=500, sir_temp=1, sir_zero_weight_exponent=-700,
                                     show_progress=True)
    elif planner_name.lower() == "refsolver-nips":
        planner = pomdp_py.RefSolverNIPS(planning_time=planning_time, num_sims=num_sims,
                                        max_depth=50, rollout_depth=50,
                                        discount_factor=discount_factor, reward_scaling_factor=1/1000,
                                        num_sir_particles=500, sir_temp=1, sir_zero_weight_exponent=-700,
                                        show_progress=True)
    elif planner_name.lower() == "refpol":
        planner = pomdp_py.RefPol(discount_factor=discount_factor,
                                  num_sir_particles=500, sir_temp=1, sir_zero_weight_exponent=-700,
                                  reseed_global_objective=True)
    else:
        planner = pomdp_py.POMCP(planning_time=planning_time, num_sims=num_sims,
                                 max_depth=50, discount_factor=discount_factor, exploration_const=np.sqrt(2),
                                 num_visits_init=0, value_init=0,
                                 rollout_policy=mp_problem.agent.policy_model,
                                 show_progress=True)
        if not planner_name.lower() == "pomcp":
            print("Unrecognized planner name. Using default planner.")

    # Step 4. Run experiments.
    run_experiments(problem=mp_problem, planner=planner, runs=runs, primitive_steps=primitive_steps,
                    file_logging=file_logging, prm_logging=prm_logging,
                    log_directory=str(pathlib.Path(__file__).parent.parent /
                                      f'experiment_logs/{env_name}_{planner_name}_'
                                      f'{time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())}'
                                      f'_pt_{planning_time}'),
                    visualize_world=visualize_world, visualize_belief=visualize_belief,
                    env_evolution_steps=env_evolution_steps)

if __name__ == '__main__':
    planning_time = 10
    main(planner_name="porpi", env_name="corsica",
         planning_time=planning_time, num_sims=-1,
         runs=10, primitive_steps=500,
         discount_factor=0.99,
         file_logging=True, plot_prm=False,
         visualize_sims=True, visualize_belief=False,
         max_macro_action_length=15,
         env_evolution_steps=[15, 50, 200])