import pomdp_py
import pathlib
from pomdp_py.problems.motion_planning.holonomic_rescue.holonomic_rescue import *
from pomdp_py.problems.motion_planning.environments.corsica import Corsica
from pomdp_py.problems.motion_planning.test_utils.log_replayer import LogReplayer

def instantiate_pomdp():

    # Step 1. Instantiate the motion planning environment.
    candidate_init_positions = [(-100, 0, 8)]
    init_pos = (-100, 0, 8)

    pyb_env = Corsica(gui=False, debugger=False, dz_level=0)
    pyb_env_gui = Corsica(gui=True, debugger=False, dz_level=0)

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
                       goal=pyb_env.goal_checker(pos+(0,0,0))) for pos in candidate_init_positions]*100)

    problem = HolonomicRescuePOMDP(init_state, init_belief,
                                  pyb_env, pyb_env_gui,
                                  max_macro_action_length=3,
                                  prm_nodes=100,
                                  prm_lower_bounds=(-100, -100, 0),
                                  prm_upper_bounds=(100, 100, 50))

    return problem

if __name__ == '__main__':
    problem = instantiate_pomdp()
    log_replayer = LogReplayer(problem)
    prefix = str(pathlib.Path(__file__).parent.parent) + "/experiment_logs/"
    solver = "porpi"
    date = "2025-01-22"
    time_stamp = "16:58:13"
    pt = 10
    runs = list(range(1, 101))
    log_files = [prefix + f"corsica_{solver}_{date}_{time_stamp}_pt_{pt}/{date} {time_stamp}_run_{run}.log" for run in runs]

    log_replayer.replay_logs(log_files, belief=False, delay=0., step_thru=False, env_evolution_steps=[], bel_timeout=10, bespoke_parser="holonomic_rescue")
    # log_replayer.replay_logs(log_files, belief=True, step_thru=True, env_evolution_steps=[], bel_timeout=2)
    # log_replayer.plot_prm(prefix + f"corsica_{solver}_{date}_{time_stamp}_pt_{pt}/{date} {time_stamp}_prm.log")