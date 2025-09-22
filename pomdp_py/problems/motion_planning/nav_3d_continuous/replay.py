import pomdp_py
import pathlib

from pomdp_py.problems.motion_planning.nav_3d_continuous.nav_3d_continuous import *
from pomdp_py.problems.motion_planning.environments.maze_3d import Maze3D
from pomdp_py.problems.motion_planning.test_utils.log_replayer import LogReplayer

def instantiate_pomdp(start=1):

    # Step 1. Instantiate the motion planning environment.
    candidate_init_positions = [(-28, 15, 0),
                                (-28, -11, 0)]

    init_pos = (-28, 15, 0) if start == 0 else (28, -11, 0)

    pyb_env_gui = Maze3D(init_pos + (0, 0, 0), True, False)
    pyb_env = Maze3D(init_pos + (0, 0, 0), False, False)

    # Step 2. Instantiate the POMDP of interest.
    init_state = Nav3DState(init_pos,
                       danger_zone=pyb_env.dz_checker(init_pos+(0,0,0)),
                       landmark=pyb_env.lm_checker(init_pos+(0,0,0)),
                       goal=pyb_env.goal_checker(init_pos+(0,0,0)))

    init_belief = pomdp_py.Particles([Nav3DState(pos,
                       danger_zone=pyb_env.dz_checker(pos+(0,0,0)),
                       landmark=pyb_env.lm_checker(pos+(0,0,0)),
                       goal=pyb_env.goal_checker(pos+(0,0,0))) for pos in candidate_init_positions]*100)

    problem = Nav3DContinuousPOMDP(init_state, init_belief,
                                      pyb_env, pyb_env_gui,
                                      max_macro_action_length=3,
                                      prm_nodes=50, prm_assert_connected=False)
    return problem

if __name__ == '__main__':
    problem = instantiate_pomdp()
    log_replayer = LogReplayer(problem)
    prefix = str(pathlib.Path(__file__).parent.parent) + "/experiment_logs/"
    # prefix = str(pathlib.Path(__file__).parent.parent) + "/experiment_logs/elmo/"
    # prefix = str(pathlib.Path(__file__).parent.parent) + "/experiment_logs/oscar/"
    solver = "porpi"
    date = "2025-01-22"

    start = 0
    time_stamp = "17:24:39"
    pt = 2

    sims = 100
    runs = list(range(1,51))
    log_files = [prefix + f"maze_3d_{start}_{solver}_{date}_{time_stamp}_pt_{pt}/{date} {time_stamp}_run_{run}.log" for run in runs]

    log_replayer.replay_logs(log_files, belief=False, delay=0., step_thru=False, env_evolution_steps=[], bel_timeout=5, bel_viz_life_time=1., bespoke_parser="maze_3d")
    # log_replayer.plot_prm(prefix + f"maze_3d_{start}_{solver}_{date}_{time_stamp}_pt_{pt}/{date} {time_stamp}_prm_run_3.log")