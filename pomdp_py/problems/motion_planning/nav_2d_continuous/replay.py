import pomdp_py
import pathlib

from pomdp_py.problems.motion_planning.nav_2d_continuous.nav_2d_continuous import *
from pomdp_py.problems.motion_planning.environments.maze_2d import Maze2D
from pomdp_py.problems.motion_planning.test_utils.log_replayer import LogReplayer

def instantiate_pomdp(start=1):

    # Step 1. Instantiate the motion planning environment.
    candidate_init_positions = [(1, -13),
                                (1, -40)]

    init_pos = (1, -13) if start == 0 else (1, -40)

    pyb_env_gui = Maze2D(init_pos + (0, 0, 0, 0), 0.0, True)
    pyb_env = Maze2D(init_pos + (0, 0, 0, 0), 0.0, False)

    init_state = Nav2DState(init_pos,
                            danger_zone=pyb_env.dz_checker(init_pos + (0, 0, 0, 0)),
                            landmark=pyb_env.lm_checker(init_pos + (0, 0, 0, 0)),
                            goal=pyb_env.goal_checker(init_pos + (0, 0, 0, 0)))

    init_belief = pomdp_py.Particles([Nav2DState(pos,
                                                 danger_zone=pyb_env.dz_checker(pos + (0, 0, 0, 0)),
                                                 landmark=pyb_env.lm_checker(pos + (0, 0, 0, 0)),
                                                 goal=pyb_env.goal_checker(pos + (0, 0, 0, 0))) for pos in
                                      candidate_init_positions] * 100)

    problem = Nav2DContinuousPOMDP(init_state, init_belief,
                                      pyb_env, pyb_env_gui,
                                      max_macro_action_length=5)
    return problem

if __name__ == '__main__':
    start = 0
    problem = instantiate_pomdp(start=start)
    log_replayer = LogReplayer(problem)
    prefix = str(pathlib.Path(__file__).parent.parent) + "/experiment_logs/"
    solver = "porpi"
    date = "2025-01-17"
    time_stamp = "12:24:06"
    pt = 30.0
    sims = 100
    runs = list(range(1,51))
    # runs = [17, 21, 29, 37, 50]
    # runs = [1, 16, 32, 38]
    # runs = [29]
    log_files = [prefix + f"maze_2d_{start}_{solver}_{date}_{time_stamp}_pt_{pt}/{date} {time_stamp}_run_{run}.log" for run in runs]

    log_replayer.replay_logs(log_files, belief=False, delay=0., step_thru=True, env_evolution_steps=[], bel_timeout=5, bel_viz_life_time=1.)
    # log_replayer.plot_prm(prefix + f"maze_2d_{start}_{solver}_{date}_{time_stamp}_pt_{pt}/{date} {time_stamp}_prm.log")