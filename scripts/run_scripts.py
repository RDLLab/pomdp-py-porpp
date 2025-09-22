import argparse

mp_problems = ["nav_3d_continuous", "nav_2d_continuous", "holonomic_rescue", "holonomic_rescue_stress"]

def parse_args():
    parser = argparse.ArgumentParser(description="pomdp_py CLI")
    parser.add_argument("--problem_name", type=str, default="nav_3d_continuous")
    parser.add_argument("--planner_name", type=str, default="ref_solver_fast")
    parser.add_argument("--env_name", type=str, default="terrain")
    parser.add_argument("--planning_time", type=float, default=1.0)
    parser.add_argument("--num_sims", type=int, default=-1)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--primitive_steps", type=int, default=500)
    parser.add_argument("--file_logging", type=bool, default=True)
    parser.add_argument("--prm_logging", type=bool, default=True)
    parser.add_argument("--plot_prm", type=bool, default=False)
    parser.add_argument("--visualize_sims", type=bool, default=False)
    parser.add_argument("--visualize_world", type=bool, default=True) # visualize true world
    parser.add_argument("--visualize_belief", type=bool, default=False)
    parser.add_argument("--max_macro_action_length", type=int, default=1)
    parser.add_argument("--start", type=int, default=0) # id of true start state
    args = parser.parse_args()
    return parser, args

if __name__ == "__main__":
    parser, args = parse_args()
    if args.problem_name.lower() == "nav_3d_continuous":
        if args.planner_name == "pomcp": # TODO: Currently no functionality available to reseed rollout policy.
            from pomdp_py.problems.motion_planning.nav_3d_continuous.problem import main
        else:
            from pomdp_py.problems.motion_planning.nav_3d_continuous.problem_reseeded import main
        main(planner_name=args.planner_name,
             env_name=args.env_name,
             planning_time=args.planning_time,
             num_sims=args.num_sims,
             runs=args.runs,
             primitive_steps=args.primitive_steps,
             file_logging=args.file_logging,
             prm_logging=args.prm_logging,
             plot_prm=args.plot_prm,
             visualize_sims=args.visualize_sims,
             visualize_world=args.visualize_world,
             visualize_belief=args.visualize_belief,
             max_macro_action_length=args.max_macro_action_length,
             start=args.start)

    elif args.problem_name.lower() == "nav_2d_continuous":
        from pomdp_py.problems.motion_planning.nav_2d_continuous.problem import main
        main(planner_name=args.planner_name,
             env_name=args.env_name,
             planning_time=args.planning_time,
             num_sims=args.num_sims,
             runs=args.runs,
             primitive_steps=args.primitive_steps,
             file_logging=args.file_logging,
             prm_logging=args.prm_logging,
             plot_prm=args.plot_prm,
             visualize_sims=args.visualize_sims,
             visualize_world=args.visualize_world,
             visualize_belief=args.visualize_belief,
             max_macro_action_length=args.max_macro_action_length,
             start=args.start)

    elif args.problem_name.lower() == "holonomic_rescue":
        from pomdp_py.problems.motion_planning.holonomic_rescue.problem import main
        main(planner_name=args.planner_name,
             env_name=args.env_name,
             planning_time=args.planning_time,
             num_sims=args.num_sims,
             runs=args.runs,
             primitive_steps=args.primitive_steps,
             file_logging=args.file_logging,
             prm_logging=args.prm_logging,
             plot_prm=args.plot_prm,
             visualize_sims=args.visualize_sims,
             visualize_world=args.visualize_world,
             visualize_belief=args.visualize_belief,
             max_macro_action_length=args.max_macro_action_length)
             
    elif args.problem_name.lower() == "holonomic_rescue_stress":
        from pomdp_py.problems.motion_planning.holonomic_rescue_stress.problem import main
        main(planner_name=args.planner_name,
             env_name=args.env_name,
             planning_time=args.planning_time,
             num_sims=args.num_sims,
             runs=args.runs,
             primitive_steps=args.primitive_steps,
             file_logging=args.file_logging,
             prm_logging=args.prm_logging,
             plot_prm=args.plot_prm,
             visualize_sims=args.visualize_sims,
             visualize_world=args.visualize_world,
             visualize_belief=args.visualize_belief,
             max_macro_action_length=args.max_macro_action_length)
             
    else:
        raise ValueError(f"Unrecognized problem name: {args.problem_name}. Available options: {mp_problems}.")
