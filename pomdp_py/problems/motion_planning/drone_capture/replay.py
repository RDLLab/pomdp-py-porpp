import pomdp_py
import pathlib

from pomdp_py.problems.motion_planning.drone_capture.drone_capture import *
from pomdp_py.problems.motion_planning.environments.drone_capture import DroneCapture
from pomdp_py.problems.motion_planning.test_utils.log_replayer_drone_capture import LogReplayer

def instantiate_pomdp(start=1):

    # Step 1. Instantiate the motion planning environment.
    predator_init_positions = DroneCapture.PREDATOR_INIT_POSITIONS
    prey_init_positions = DroneCapture.PREY_INIT_POSITIONS
    captured = [0]

    pyb_env_gui = DroneCapture(gui=True, debugger=False)
    pyb_env = DroneCapture(gui=False, debugger=False)

    # Step 2. Instantiate the POMDP of interest.
    init_state = PPState(predator_positions=predator_init_positions,
                         prey_positions=DroneCapture.PREY_INIT_POSITIONS,
                         captured=captured)

    init_belief = pomdp_py.Particles([PPState(predator_positions=DroneCapture.PREDATOR_INIT_POSITIONS,
                                            prey_positions=pyb_env.sample_free_prey_config(),
                                            captured=[0])
                                    for i in range(100)])


    problem = DroneCapturePOMDP(init_state, init_belief,
                                      pyb_env, pyb_env_gui)
    return problem

if __name__ == '__main__':
    problem = instantiate_pomdp()
    log_replayer = LogReplayer(problem)
    prefix = str(pathlib.Path(__file__).parent.parent) + "/experiment_logs/capture_logs/"
    date = "2024-08-05"
    time_stamp = "15:34:25"
    # runs = [21]
    # runs = [19] # OK
    # runs = [30] # NICE
    # runs = [15] # NICE

    runs = list(range(1, 51))

    log_files = [prefix + f"{date} {time_stamp}_run_{run}.log" for run in runs]

    log_replayer.replay_logs(log_files, belief=True, delay=0., step_thru=True, env_evolution_steps=[],
                             trace_trajectory=True, trajectory_width=0.01,
                             bel_timeout=20, bespoke_parser="drone_capture", bel_viz_life_time=1.)
