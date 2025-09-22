import argparse
import pathlib
import time
import typing
import re
import sys

import jsonpickle
import pybullet as pyb
import copy

class LogReplayer:
    BELIEF_PATTERN =           "Belief (json):       "
    BELIEF_HUMAN_PATTERN =     "Belief:              "
    STATE_PATTERN =            "True State (json):   "
    STATE_HUMAN_PATTERN =      "True State:          "
    ACTION_PATTERN =           "Action:              "
    NEXT_STATE_PATTERN =       "Next State (json):   "
    NEXT_STATE_HUMAN_PATTERN = "Next State:          "
    OBSERVATION_PATTERN =      "Observation:         "
    TOTAL_STEPS_PATTERN =      "Total Steps:         "
    REWARD_PATTERN =           "Reward:              "
    CUM_DISC_REWARD_PATTERN =  "Cum. Disc. Reward:   "
    STEP_COMPLETE_PATTERN =    "=== STEP COMPLETE ==="
    RUN_COMPLETE_PATTERN =     "=== RUN COMPLETE! ==="

    def __init__(self, problem):
        """
        Initialize the LogReplay class
        """
        self.problem = problem
        self.orig_problem = copy.deepcopy(problem)

    def next_step_data(self, fp: typing.TextIO):
        line = fp.readline()
        end_reached = not line or line.strip() == self.RUN_COMPLETE_PATTERN

        if end_reached:
            return None, None, None, None, None, None, None, None, None, None, None, end_reached

        b, bh, s, sh, a, ns, nsh, o, tot_prim_steps, r, cdr = None, None, None, None, None, None, None, None, None, None, None
        step_complete = False

        while not all([b, bh, s, sh, a, ns, nsh, o, tot_prim_steps, r, cdr, step_complete]) and not end_reached:
            line = fp.readline()
            if line.startswith(self.BELIEF_HUMAN_PATTERN):
                line = line.strip()
                bh = line.replace(self.BELIEF_HUMAN_PATTERN, "", 1)
            elif line.startswith(self.BELIEF_PATTERN):
                line = line.strip()
                b = line.replace(self.BELIEF_PATTERN, "", 1)
            elif line.startswith(self.STATE_HUMAN_PATTERN):
                line = line.strip()
                sh = line.replace(self.STATE_HUMAN_PATTERN, "", 1)
            elif line.startswith(self.STATE_PATTERN):
                line = line.strip()
                s = line.replace(self.STATE_PATTERN, "", 1)
            elif line.startswith(self.ACTION_PATTERN):
                line = line.strip()
                a = line.replace(self.ACTION_PATTERN, "", 1)
            elif line.startswith(self.NEXT_STATE_HUMAN_PATTERN):
                line = line.strip()
                nsh = line.replace(self.NEXT_STATE_HUMAN_PATTERN, "", 1)
            elif line.startswith(self.NEXT_STATE_PATTERN):
                line = line.strip()
                ns = line.replace(self.NEXT_STATE_PATTERN, "", 1)
            elif line.startswith(self.OBSERVATION_PATTERN):
                line = line.strip()
                o = line.replace(self.OBSERVATION_PATTERN, "", 1)
            elif line.startswith(self.TOTAL_STEPS_PATTERN):
                line = line.strip()
                tot_prim_steps = line.replace(self.TOTAL_STEPS_PATTERN, "", 1)
            elif line.startswith(self.REWARD_PATTERN):
                line = line.strip()
                r = line.replace(self.REWARD_PATTERN, "", 1)
            elif line.startswith(self.CUM_DISC_REWARD_PATTERN):
                line = line.strip()
                cdr = line.replace(self.CUM_DISC_REWARD_PATTERN, "", 1)
            elif line.startswith(self.STEP_COMPLETE_PATTERN):
                step_complete = True
            elif line.startswith(self.RUN_COMPLETE_PATTERN) or line.startswith("Exception"):
                end_reached = True
                break

        return b, bh, s, sh, a, ns, nsh, o, tot_prim_steps, r, cdr, end_reached

    def replay_log(self, log_file,
                   start_on_user_prompt=True, delay=1., step_thru=True,
                   trace_trajectory=True, trajectory_rgb=(0,0,0), trajectory_width=2.,
                   belief=True, bel_timeout=10., bel_viz_life_time=0.5,
                   env_evolution_steps=[],
                   bespoke_parser=None):
        """
        Replay log_file from a given path.

        log_file: A string or a pathlib.Path instance specifying the file.
        start_on_user_prompt (bool): If True, only starts replay after user prompt.
        delay (int): The delay between replay steps.
        step_thru (bool): If True, step through the log file upon user input.
            Otherwise, step through automatically based at the delay speed.
        trace_trajectory (bool): If True, traces the executed trajectory.
        trajectory_rgb (tuple): The RGB color of the trajectory.
        trajectory_width (float): The width of the trajectory line.
        belief (bool): If True, the belief of the log file will be printed to console and GUI.
        bel_timeout (flot): The timeout before the GUI stops plotting belief particles.
        bel_viz_life_time (float): The GUI lifetime of the belief visualization.
        env_evolution_steps (list): If the experiments have evolving environments, a list
            of steps at which the environment evolves.
        bespoke_parser (list): A string for the bespoke parser when only the human-readable part
            of the log file is available. Default: None.
        """

        if isinstance(log_file, str):
            log_file_path = pathlib.Path(log_file)
        elif isinstance(log_file, pathlib.Path):
            log_file_path = log_file
        else:
            raise TypeError("Parameter log_file should be a 'string' or an instance of 'pathlib.Path'.")

        if not log_file_path.exists():
            raise FileNotFoundError(f"Can not find the file {log_file_path}.")

        if start_on_user_prompt:
            input(f"Press <Enter> to replay log file {log_file_path.name}.")

        step = 1
        env_index = 0
        prev_s = None
        with log_file_path.open() as fp:
            while True:
                b, bh, s, sh, a, ns, nsh, o, total_steps, r, cdr, end_reached = self.next_step_data(fp)

                if end_reached:
                    print("Reached end of log file.\n")
                    time.sleep(0.2)
                    input("Press <Enter> to continue...")
                    break

                # Update environment if applicable.
                # TODO: Make this more robust.
                if len(env_evolution_steps):
                    if int(total_steps) >= env_evolution_steps[env_index] and env_index < len(env_evolution_steps):
                        env_index += 1
                        self.problem.update_environment(env_index)

                if bespoke_parser is None:
                    # Decode pickled data
                    b = jsonpickle.decode(b)
                    s = jsonpickle.decode(s)
                    ns = jsonpickle.decode(ns)
                elif bespoke_parser == "maze_3d":
                    from pomdp_py.problems.motion_planning.nav_3d_continuous.nav_3d_continuous import Nav3DState
                    bel_positions = eval(re.sub(" \| dz: [A-Za-z]* \| lm: [A-Za-z]* \| goal: [A-Za-z]*>:", ":", re.sub("<pos: ", "", bh)))
                    b = {Nav3DState(pos, False, False, False): p for pos, p in bel_positions.items()} # Doesn't matter what DZ, LM and Goal are.
                    s_xyz = eval(re.sub("\).*", ")", re.sub("<pos: ", "", sh)))
                    s_dz = eval(re.findall("dz: [A-Za-z]*", sh)[0].replace("dz: ", ""))
                    s_lm = eval(re.findall("lm: [A-Za-z]*", sh)[0].replace("lm: ", ""))
                    s_goal = eval(re.findall("goal: [A-Za-z]*", sh)[0].replace("goal: ", ""))
                    ns_xyz = eval(re.sub("\).*", ")", re.sub("<pos: ", "", nsh)))
                    ns_dz = eval(re.findall("dz: [A-Za-z]*", nsh)[0].replace("dz: ", ""))
                    ns_lm = eval(re.findall("lm: [A-Za-z]*", nsh)[0].replace("lm: ", ""))
                    ns_goal = eval(re.findall("goal: [A-Za-z]*", nsh)[0].replace("goal: ", ""))
                    s = Nav3DState(s_xyz, s_dz, s_lm, s_goal)
                    ns = Nav3DState(ns_xyz, ns_dz, ns_lm, ns_goal)
                elif bespoke_parser == "holonomic_rescue":
                    from pomdp_py.problems.motion_planning.holonomic_rescue.holonomic_rescue import Nav3DState
                    if belief:
                        bel_positions = eval(re.sub(" \| col: [A-Za-z]* \| nfz: [A-Za-z]* \| lm: [A-Za-z]* \| objectives_reached: \[[^\|]*\] \| mission_accomplished: [A-Za-z]*>:", ":", re.sub("<pos: ", "", bh)))
                        b = {Nav3DState(pos, collision=False, no_fly_zone=False, landmark=False, objectives_reached=[], goal=False): p for pos, p in bel_positions.items()} # Doesn't matter what col, nfz, lm, obj_r, goal are.
                    s_xyz = eval(re.sub("\).*", ")", re.sub("<pos: ", "", sh)))
                    s_col = eval(re.findall("col: [A-Za-z]*", sh)[0].replace("col: ", ""))
                    s_nfz = eval(re.findall("nfz: [A-Za-z]*", sh)[0].replace("nfz: ", ""))
                    s_lm = eval(re.findall("lm: [A-Za-z]*", sh)[0].replace("lm: ", ""))
                    s_obj = eval(re.findall("objectives_reached: \[[^\|]*\]", sh)[0].replace("objectives_reached: ", ""))
                    s_goal = eval(re.findall("mission_accomplished: [A-Za-z]*", sh)[0].replace("mission_accomplished: ", ""))
                    ns_xyz = eval(re.sub("\).*", ")", re.sub("<pos: ", "", nsh)))
                    ns_col = eval(re.findall("col: [A-Za-z]*", nsh)[0].replace("col: ", ""))
                    ns_nfz = eval(re.findall("nfz: [A-Za-z]*", nsh)[0].replace("nfz: ", ""))
                    ns_lm = eval(re.findall("lm: [A-Za-z]*", nsh)[0].replace("lm: ", ""))
                    ns_obj = eval(re.findall("objectives_reached: \[[^\|]*\]", nsh)[0].replace("objectives_reached: ", ""))
                    ns_goal = eval(re.findall("mission_accomplished: [A-Za-z]*", nsh)[0].replace("mission_accomplished: ", ""))
                    s = Nav3DState(s_xyz, s_col, s_nfz, s_lm, s_obj, s_goal)
                    ns = Nav3DState(ns_xyz, ns_col, ns_nfz, ns_lm, ns_obj, ns_goal)
                elif bespoke_parser == "drone_capture":
                    from pomdp_py.problems.motion_planning.drone_capture.drone_capture import PPState
                    from pomdp_py.problems.motion_planning.environments.drone_capture import tuple_list_to_list
                    bel_prey_positions = eval(re.sub("\]", "", re.sub("<prey: \[", "", re.sub(" \| pred: [\[(-.,0-9\]\s]* \| captured: [\[01\]]*>", "", bh))))
                    b = {PPState([0]*12, pos, [0]): p for pos,p in bel_prey_positions.items()}
                    s_predators = eval(re.sub(" \| captured: \[[01]*\]>", "", re.sub("<prey: \[[0-9,.)(\s]*\] \| pred: ", "", sh)))
                    s_predators = tuple_list_to_list(s_predators)
                    s_prey = list(eval(re.sub("<prey: \[", "", re.sub("\] \| pred:.*", "", sh))))
                    s_captured = re.sub(">", "", re.sub(".* \| captured: ", "", sh))

                    ns_predators = eval(re.sub(" \| captured: \[[01]*\]>", "", re.sub("<prey: \[[0-9,.)(\s]*\] \| pred: ", "", nsh)))
                    ns_predators = tuple_list_to_list(ns_predators)
                    ns_prey = list(eval(re.sub("<prey: \[", "", re.sub("\] \| pred:.*", "", nsh))))
                    ns_captured = re.sub(">", "", re.sub(".* \| captured: ", "", nsh))

                    s = PPState(s_predators, s_prey, s_captured)
                    ns = PPState(ns_predators, ns_prey, ns_captured)
                else:
                    raise ValueError("Unexpected environment identifier.")

                print(f"=== STEP {step} ===")
                print(f"True State:          {s}")
                print(f"{self.ACTION_PATTERN + a}")
                print(f"Next State:          {ns}")
                print(f"{self.OBSERVATION_PATTERN + o}")
                print(f"{self.TOTAL_STEPS_PATTERN + total_steps}")
                print(f"{self.REWARD_PATTERN + r}")
                print(f"{self.CUM_DISC_REWARD_PATTERN + cdr}")

                self.problem.env.apply_transition(s)
                self.problem.visualize_world()

                if trace_trajectory:
                    if prev_s is not None:
                        pyb.addUserDebugLine(lineFromXYZ=prev_s.xyz, lineToXYZ=s.xyz,
                                         lineColorRGB=trajectory_rgb, lineWidth=trajectory_width, lifeTime=0,
                                         physicsClientId=self.problem._pyb_env_gui._id)
                    prev_s = s

                if belief:
                    if bespoke_parser is None:
                        bel_histogram = {jsonpickle.decode(s): p for s, p in b.items()}
                    else:
                        bel_histogram = b
                    print(f"Belief:              {bel_histogram}")
                    self.problem.visualize_belief(bel_histogram, life_time=bel_viz_life_time, timeout=bel_timeout)

                # if trace_trajectory:
                #     if prev_s is not None:
                #         pyb.addUserDebugLine(lineFromXYZ=prev_s.xyz, lineToXYZ=s.xyz,
                #                          lineColorRGB=trajectory_rgb, lineWidth=trajectory_width, lifeTime=0,
                #                          physicsClientId=self.problem._pyb_env_gui._id)
                #     prev_s = s

                step += 1
                if step_thru:
                    input("Press <Enter> to continue...")
                else:
                    time.sleep(delay)

    def replay_logs(self, log_files,
                    start_on_user_prompt=True, delay=1., step_thru=True,
                    trace_trajectory=True, trajectory_rgb=(0, 0, 0), trajectory_width=2.,
                    belief=True, bel_timeout=10., bel_viz_life_time=0.5,
                    env_evolution_steps=[],
                    bespoke_parser=None,
                    retain_trajectories=False):
        """
        Replay all the logs from given log files list.

        log_files: A single log_file or a list of log_files.
        start_on_user_prompt (bool): If True, only starts replay after user prompt.
        delay (int): The delay between replay steps.
        step_thru (bool): If True, step through the log file upon user input.
            Otherwise, step through automatically based at the delay speed.
        trace_trajectory (bool): If True, traces the executed trajectory.
        trajectory_rgb (tuple): The RGB color of the trajectory.
        trajectory_width (float): The width of the trajectory line.
        belief (bool): If True, the belief of the log file will be printed to console and GUI.
        bel_timeout (flot): The timeout before the GUI stops plotting belief particles.
        bel_timeout (flot): The timeout before the GUI stops plotting belief particles.
        bel_viz_life_time (float): The GUI lifetime of the belief visualization.
        env_evolution_steps (list): If the experiments have evolving environments, a list
            of steps at which the environment evolves.
        bespoke_parser (list): A string for the bespoke parser when only the human-readable part
            of the log file is available. Default: None.
        retain_trajectories (bool): If True, cumulatively retain visualized trajectories across runs.
        """

        if not isinstance(log_files, list):
            log_files = [log_files]

        for log_file in log_files:
            self.replay_log(log_file,
                            start_on_user_prompt, delay, step_thru,
                            trace_trajectory, trajectory_rgb, trajectory_width,
                            belief, bel_timeout, bel_viz_life_time,
                            env_evolution_steps, bespoke_parser)

            self.reset_replay_environment(retain_trajectories)

    def reset_replay_environment(self, retain_trajectories=False):
        self.problem = self.orig_problem
        if not retain_trajectories:
            pyb.removeAllUserDebugItems(physicsClientId=self.problem._pyb_env_gui._id)

    def plot_prm(self, prm_log_file, line_width=0.1, line_color=(0.2, 0.2, 0.2)):
        if isinstance(prm_log_file, str):
            prm_log_file = pathlib.Path(prm_log_file)
        else:
            if not isinstance(prm_log_file, pathlib.Path):
                raise TypeError("Parameter prm_log_file should be a 'string' or an instance of 'pathlib.Path'.")

        for line in prm_log_file.read_text(encoding='utf-8').splitlines():
            if line.startswith('PRM Edges: '):
                edges = eval(re.search(r"\[(.*?)\]", line).group(0))
        for e in edges:
            pyb.addUserDebugLine(lineFromXYZ=e[0][0:3], lineToXYZ=e[1][0:3], lineColorRGB=line_color,
                                 lineWidth=line_width, lifeTime=0, physicsClientId=self.problem._pyb_env_gui._id)
        time.sleep(60)

    #TODO: Code below is Yohan's previous code. Not sure how compatible it is.

    @staticmethod
    def read_file_path_from_sys_argv():
        working_dir = pathlib.Path.cwd()
        if len(sys.argv) < 2:
            raise ValueError("No logfile argument")
        return (working_dir / sys.argv[1]).resolve()

    @staticmethod
    def read_all_files_from_sys_argv():
        working_dir = pathlib.Path.cwd()
        if len(sys.argv) < 2:
            raise ValueError("No log folder argument")
        log_dir = (working_dir / sys.argv[1]).resolve()
        if len(sys.argv) < 3:
            print("No logfile match string argument, using all")
            regex = r"^.+_run_\d+.log$"
        else:
            print(f"Logfile match string: {sys.argv[2]}")
            regex = rf"^{re.escape(sys.argv[2])}_run_\d+.log$"
        log_files = [entry for entry in log_dir.iterdir() if entry.is_file() and bool(re.search(regex, entry.name))]

        print(f'\nFound {len(log_files)} log files\n')

        log_files.sort()

        return log_files

    @staticmethod
    def read_from_summary_logfile():
        parser = argparse.ArgumentParser()
        parser.add_argument('summary_file', help='Summary logfile')
        parser.add_argument('-t', '--type', choices=['all', 'suc', 'fail', 'err', 'tout'], default='suc', help='Replaying run type')
        parser.add_argument('-u', '--user', type=int, choices=[0, 1], default=1, help='Wait for user input')

        args = parser.parse_args()

        working_dir = pathlib.Path.cwd()
        summary_file = working_dir / args.summary_file
        if args.user:
            user = True
        else:
            user = False
        if not summary_file.exists() and not summary_file.is_file():
            raise ValueError("Summary file does not exist")

        log_folder = summary_file.parent
        log_type = args.type

        log_line_dic = {
            'suc': ['Success runs'],
            'fail': ['Failed runs'],
            'err': ['Error runs'],
            'tout': ['Timeout runs'],
            'all': ['Success runs', 'Failed runs', 'Timeout runs', 'Error runs'],
        }

        # Extract the run numbers
        run_number_list = []

        with open(summary_file, 'r') as f:
            for line in f:
                for log_pattern in log_line_dic[log_type]:
                    if line.startswith(log_pattern):
                        s_idx = line.find('[')  # Find the list start index
                        matches = re.findall(r"\d+", line[s_idx:])
                        run_number_list.extend([int(x) for x in matches])

        # Create the file name list
        file_name_pattern_regex = r"(.*?)_summary\.log"
        match = re.search(file_name_pattern_regex, summary_file.name)
        if match is None:
            raise ValueError("Error while extracting summary log file pattern")
        file_name_pattern = match.group(1)

        log_files = []

        for run_number in run_number_list:
            run_file = log_folder / f'{file_name_pattern}_run_{run_number}.log'
            if run_file.exists() and run_file.is_file():
                log_files.append(run_file)

        return log_files, user
