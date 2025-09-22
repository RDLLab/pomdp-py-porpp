import logging
import scipy.stats as st
import numpy as np
import time
import copy
import jsonpickle
import pybullet as pyb
from datetime import datetime

import pomdp_py
from pomdp_py.framework.basics import MPPOMDP
from pomdp_py.framework.planner import Planner
from .logger import LogHelper

def get_mean_std_ci(data, confidence=0.95):
    mean = np.mean(data)
    std = np.std(data)
    ci = st.t.interval(alpha=0.95, df=len(data) - 1, loc=np.mean(data), scale=st.sem(data))

    return mean, std, ci

def get_sum(histogram):
    sum = 0.0
    for x, v in histogram.items():
        sum += v
    return sum

def run_experiments(problem: MPPOMDP, planner: Planner, runs=1, primitive_steps=100, **kwargs):
    """Runs multiple runs on a given problem and summarises results"""

    rewards = {}
    rewards_discounted = {}
    steps_dict = {}
    planning_times = {}
    num_sims = {}
    success_list = []
    failed_list = []
    errors_list = []
    runtime = {}
    error_dict = {}

    exp_time = str(datetime.now().replace(microsecond=0))
    file_logging = kwargs.get("file_logging", False)
    log_directory = kwargs.get("log_directory", None)
    prm_logging = kwargs.get("prm_logging", True)
    env_evolution_steps = kwargs.get("env_evolution_steps", [])
    seeds = kwargs.get("seeds", list(range(runs)))

    if prm_logging:
        prm_logger_name = f"{exp_time}_prm"
        prm_logger = LogHelper.get_logger(prm_logger_name, file_logging, prm_logger_name, log_directory)
        g = problem.agent.policy_model._prm.prm
        prm_logger.info(f"PRM Edges: {g.edges}")
        prm_logger.info(f"PRM Nodes: {g.nodes}")

    exp_logger_name = f"{exp_time}_summary"
    exp_logger = LogHelper.get_logger(exp_logger_name, file_logging, exp_logger_name, log_directory)

    exp_logger.info("=======================================")
    exp_logger.info(f"Starting Test Runner")
    exp_logger.info(f"Start Time: {exp_time}")
    exp_logger.info(f"Problem: {problem}")
    exp_logger.info(f"Planner: {planner}")

    for run in range(runs):
        exp_logger.info(f"Starting Run {run + 1}")
        LogHelper.setup_base_logger(file_logging, log_directory, f'{exp_time}_run_{run + 1}')

        run_start_time = time.time()

        _problem = copy.deepcopy(problem)
        if len(env_evolution_steps):
            _problem.reset_environment()

        try:
            total_reward, total_reward_discounted, total_steps, success, mean_planning_time, mean_num_sims \
                = run_experiment(_problem, planner, primitive_steps, seeds[run], **kwargs)

            rewards[run + 1] = total_reward
            rewards_discounted[run + 1] = total_reward_discounted
            steps_dict[run + 1] = total_steps
            planning_times[run + 1] = mean_planning_time
            num_sims[run + 1] = mean_num_sims
            if success:
                success_list.append(run + 1)
            else:
                failed_list.append(run + 1)
            runtime[run + 1] = time.time() - run_start_time
            exp_logger.info(f"Run {run + 1} is done)\n"
                            f"Total Discounted Reward: {total_reward_discounted} | Total Reward: {total_reward_discounted} | Success: {success}")
            exp_logger.info(f"=== RUN COMPLETE! ===")

        except Exception as ex:
            logging.exception(f"Exception in run {run + 1}")
            errors_list.append(run + 1)
            error_dict[run + 1] = ex
            if runs == 1:
                raise ex

        """
        Wait some time to finish logging, so logs don't overlap.
        """
        time.sleep(3)

    # Clear log handlers in main logger
    LogHelper.clear_log_handlers()

    exp_logger.info("=======================================")

    if runs < 30:
        exp_logger.info("WARNING: Sample size is less than 30. CI approximation may be inaccurate.")

    if runs > 1:
        exp_logger.info("---------Experiment Statistics---------")
        exp_logger.info(f"Number of Runs        : {runs}")
        exp_logger.info("-----------------------")

        # Exit if there's only one successful run
        if len(rewards) < 2:
            exp_logger.info(f"Only 1 run is successful")
            return

        reward_mean, reward_std, reward_ci = get_mean_std_ci(list(rewards.values()))
        reward_d_mean, reward_d_std, reward_d_ci = get_mean_std_ci(list(rewards_discounted.values()))
        steps_mean = np.mean(list(steps_dict.values()))
        success_ratio_all = len(success_list) / runs
        if len(errors_list) == runs:
            completed_runs = 0
            success_ratio_comp = 0.0
        else:
            completed_runs = (runs - len(errors_list))
            success_ratio_comp = len(success_list) / completed_runs
        runtime_mean = np.mean(list(runtime.values()))
        runtime_total = np.sum(list(runtime.values()))

        exp_logger.info(f"Reward Mean           : {reward_mean:.3f}")
        exp_logger.info(f"Reward SD             : {reward_std:.3f}")
        exp_logger.info(f"Reward CI             : {reward_ci}")
        exp_logger.info(f"Reward All            : {rewards}")
        exp_logger.info("-----------------------")
        exp_logger.info(f"Reward Mean (Disc.)   : {reward_d_mean:.3f}")
        exp_logger.info(f"Reward SD (Disc.)     : {reward_d_std:.3f}")
        exp_logger.info(f"Reward CI (Disc.)     : {reward_d_ci}")
        exp_logger.info(f"Reward All (Disc.)    : {rewards_discounted}")
        exp_logger.info("-----------------------")
        exp_logger.info(f"Mean Planning Times   : {planning_times}")
        exp_logger.info(f"Mean Simulations      : {num_sims}")
        exp_logger.info("-----------------------")
        exp_logger.info(f"Steps Mean            : {steps_mean:.2f}")
        exp_logger.info(f"Steps All             : {steps_dict}")
        exp_logger.info("-----------------------")
        exp_logger.info(f"Runtime Mean (sec)    : {runtime_mean:.4f}")
        exp_logger.info(f"Runtime Total (sec)   : {runtime_total:.4f}")
        exp_logger.info(f"Runtime All (sec)     : {runtime}")
        exp_logger.info("-----------------------")
        exp_logger.info(f"Completed Runs        : {completed_runs}")
        exp_logger.info(f"Success Count         : {len(success_list)}")
        exp_logger.info(f"Success Runs          : {success_list}")
        exp_logger.info(f"Failed Runs           : {failed_list}")
        exp_logger.info(f"Success Rate (All)    : {100*success_ratio_all:.2f} %")
        exp_logger.info(f"Success Rate (Compl.) : {100*success_ratio_comp:.2f} %")
        if len(error_dict) > 0:
            exp_logger.info("-----------------------")
            exp_logger.info(f"Error Count           : {len(error_dict)}")
            exp_logger.info(f"Error Runs            : {errors_list}")
            exp_logger.info(f"Errors                : {error_dict}")
        exp_logger.info("-----------------------")

def run_experiments_reseed_problem(problem_name: str, planner: Planner, runs=1, primitive_steps=100, **kwargs):
    """Runs multiple runs on a reseeded problem and summarises results"""

    rewards = {}
    rewards_discounted = {}
    steps_dict = {}
    planning_times = {}
    num_sims = {}
    success_list = []
    failed_list = []
    errors_list = []
    runtime = {}
    error_dict = {}

    exp_time = str(datetime.now().replace(microsecond=0))
    file_logging = kwargs.get("file_logging", False)
    log_directory = kwargs.get("log_directory", None)
    prm_logging = kwargs.get("prm_logging", True)
    env_evolution_steps = kwargs.get("env_evolution_steps", [])
    seeds = kwargs.get("seeds", list(range(runs)))

    exp_logger_name = f"{exp_time}_summary"
    exp_logger = LogHelper.get_logger(exp_logger_name, file_logging, exp_logger_name, log_directory)

    exp_logger.info("=======================================")
    exp_logger.info(f"Starting Test Runner")
    exp_logger.info(f"Start Time: {exp_time}")
    exp_logger.info(f"Problem: {problem_name}")
    exp_logger.info(f"Planner: {planner}")

    for run in range(runs):

        # TODO: Can this be extracted to a more abstract level?
        if problem_name.lower() == "maze_3d":
            print("Initializing Maze3D environment and POMDP")
            from pomdp_py.problems.motion_planning.environments.maze_3d import Maze3D
            from pomdp_py.problems.motion_planning.nav_3d_continuous.nav_3d_continuous import Nav3DState, Nav3DContinuousPOMDP
            start = kwargs.get("start", 0)
            max_macro_action_length = kwargs.get("max_macro_action_length", 5)
            visualize_sims = kwargs.get("visualize_sims", True)
            prm_nodes = kwargs.get("prm_nodes", 100)
            prm_assert_connected = kwargs.get("prm_assert_connected", True)

            # Step 1. Instantiate the motion planning environment.
            candidate_init_positions = [(-28, 15, 0),
                                        (-28, -11, 0)]
            if start == 0:
                init_pos = (-28, 15, 0)
            elif start == 1:
                init_pos = (-28, -11, 0)
            else:
                raise ValueError('start must be either 0 for "top" or 1 for "bottom".')

            # TODO: Can you do this with visualisation?
            pyb_env_gui = Maze3D(init_pos + (0, 0, 0), False, False)
            # pyb_env_gui = Maze3D(init_pos + (0, 0, 0), True, False)
            pyb_env = Maze3D(init_pos + (0, 0, 0), False, False)

            # Step 2. Instantiate the POMDP of interest.
            init_state = Nav3DState(init_pos,
                                    danger_zone=pyb_env.dz_checker(init_pos + (0, 0, 0)),
                                    landmark=pyb_env.lm_checker(init_pos + (0, 0, 0)),
                                    goal=pyb_env.goal_checker(init_pos + (0, 0, 0)))

            init_belief = pomdp_py.Particles([Nav3DState(pos,
                                                         danger_zone=pyb_env.dz_checker(pos + (0, 0, 0)),
                                                         landmark=pyb_env.lm_checker(pos + (0, 0, 0)),
                                                         goal=pyb_env.goal_checker(pos + (0, 0, 0))) for pos in
                                              candidate_init_positions] * 100)

            _problem = Nav3DContinuousPOMDP(init_state, init_belief,
                                              pyb_env_gui if visualize_sims else pyb_env, pyb_env_gui,
                                              max_macro_action_length=max_macro_action_length,
                                              prm_nodes=prm_nodes, prm_assert_connected=prm_assert_connected)
        else:
            raise ValueError("Unrecognized environment name.")

        if prm_logging:
            prm_logger_name = f"{exp_time}_prm_run_{run + 1}"
            prm_logger = LogHelper.get_logger(prm_logger_name, file_logging, prm_logger_name, log_directory)
            g = _problem.agent.policy_model._prm.prm
            prm_logger.info(f"PRM Edges: {g.edges}")
            prm_logger.info(f"PRM Nodes: {g.nodes}")
            time.sleep(2)

        exp_logger.info(f"Starting Run {run + 1}")
        LogHelper.setup_base_logger(file_logging, log_directory, f'{exp_time}_run_{run + 1}')

        run_start_time = time.time()

        if len(env_evolution_steps):
            _problem.reset_environment()

        try:
            total_reward, total_reward_discounted, total_steps, success, mean_planning_time, mean_num_sims \
                = run_experiment(_problem, planner, primitive_steps, seeds[run], **kwargs)

            rewards[run + 1] = total_reward
            rewards_discounted[run + 1] = total_reward_discounted
            steps_dict[run + 1] = total_steps
            planning_times[run + 1] = mean_planning_time
            num_sims[run + 1] = mean_num_sims
            if success:
                success_list.append(run + 1)
            else:
                failed_list.append(run + 1)
            runtime[run + 1] = time.time() - run_start_time
            exp_logger.info(f"Run {run + 1} is done)\n"
                            f"Total Discounted Reward: {total_reward_discounted} | Total Reward: {total_reward_discounted} | Success: {success}")
            exp_logger.info(f"=== RUN COMPLETE! ===")

        except Exception as ex:
            logging.exception(f"Exception in run {run + 1}")
            errors_list.append(run + 1)
            error_dict[run + 1] = ex
            if runs == 1:
                raise ex

        """
        Wait some time to finish logging, so logs don't overlap.
        """
        time.sleep(3)

        # Clear log handlers in main logger
        LogHelper.clear_log_handlers()

    exp_logger.info("=======================================")

    if runs < 30:
        exp_logger.info("WARNING: Sample size is less than 30. CI approximation may be inaccurate.")

    if runs > 1:
        exp_logger.info("---------Experiment Statistics---------")
        exp_logger.info(f"Number of Runs        : {runs}")
        exp_logger.info("-----------------------")

        # Exit if there's only one successful run
        if len(rewards) < 2:
            exp_logger.info(f"Only 1 run is successful")
            return

        reward_mean, reward_std, reward_ci = get_mean_std_ci(list(rewards.values()))
        reward_d_mean, reward_d_std, reward_d_ci = get_mean_std_ci(list(rewards_discounted.values()))
        steps_mean = np.mean(list(steps_dict.values()))
        success_ratio_all = len(success_list) / runs
        if len(errors_list) == runs:
            completed_runs = 0
            success_ratio_comp = 0.0
        else:
            completed_runs = (runs - len(errors_list))
            success_ratio_comp = len(success_list) / completed_runs
        runtime_mean = np.mean(list(runtime.values()))
        runtime_total = np.sum(list(runtime.values()))

        exp_logger.info(f"Reward Mean           : {reward_mean:.3f}")
        exp_logger.info(f"Reward SD             : {reward_std:.3f}")
        exp_logger.info(f"Reward CI             : {reward_ci}")
        exp_logger.info(f"Reward All            : {rewards}")
        exp_logger.info("-----------------------")
        exp_logger.info(f"Reward Mean (Disc.)   : {reward_d_mean:.3f}")
        exp_logger.info(f"Reward SD (Disc.)     : {reward_d_std:.3f}")
        exp_logger.info(f"Reward CI (Disc.)     : {reward_d_ci}")
        exp_logger.info(f"Reward All (Disc.)    : {rewards_discounted}")
        exp_logger.info("-----------------------")
        exp_logger.info(f"Mean Planning Times   : {planning_times}")
        exp_logger.info(f"Mean Simulations      : {num_sims}")
        exp_logger.info("-----------------------")
        exp_logger.info(f"Steps Mean            : {steps_mean:.2f}")
        exp_logger.info(f"Steps All             : {steps_dict}")
        exp_logger.info("-----------------------")
        exp_logger.info(f"Runtime Mean (sec)    : {runtime_mean:.4f}")
        exp_logger.info(f"Runtime Total (sec)   : {runtime_total:.4f}")
        exp_logger.info(f"Runtime All (sec)     : {runtime}")
        exp_logger.info("-----------------------")
        exp_logger.info(f"Completed Runs        : {completed_runs}")
        exp_logger.info(f"Success Count         : {len(success_list)}")
        exp_logger.info(f"Success Runs          : {success_list}")
        exp_logger.info(f"Failed Runs           : {failed_list}")
        exp_logger.info(f"Success Rate (All)    : {100*success_ratio_all:.2f} %")
        exp_logger.info(f"Success Rate (Compl.) : {100*success_ratio_comp:.2f} %")
        if len(error_dict) > 0:
            exp_logger.info("-----------------------")
            exp_logger.info(f"Error Count           : {len(error_dict)}")
            exp_logger.info(f"Error Runs            : {errors_list}")
            exp_logger.info(f"Errors                : {error_dict}")
        exp_logger.info("-----------------------")

def run_experiment(problem: MPPOMDP, planner: Planner, primitive_steps=100, seed=0, **kwargs):
    """Runs and the action-feedback loop for the POMDP and records results."""

    np.random.seed(seed=seed)

    total_reward = 0.0
    total_reward_discounted = 0.0
    total_steps = 0
    success = 0
    planning_times_per_step = []
    num_sims_per_step = []

    env_evolution_steps = kwargs.get("env_evolution_steps", [])

    i = 1
    env_index = 0
    while total_steps <= primitive_steps:

        if env_index < len(env_evolution_steps):
            if total_steps >= env_evolution_steps[env_index]:
                env_index += 1
                problem.update_environment(env_index)

        """
        Step 1. Agent plans an action...
        """
        hist = problem.agent.belief.get_histogram()

        if kwargs.get("visualize_belief"):
            problem.visualize_belief(hist, life_time=0.0, timeout=3)

        action = planner.plan(problem.agent)
        if not isinstance(planner, pomdp_py.RefPol):
            planning_times_per_step.append(planner.last_planning_time)
            num_sims_per_step.append(planner.last_num_sims)
        logging.info(f"\n\n===== STEP {i} =====")
        logging.info(f"V:                   {problem.agent.tree}")
        if (isinstance(planner, pomdp_py.PORPI)
            or isinstance(planner, pomdp_py.RefSolver)
            or isinstance(planner, pomdp_py.RefSolverNIPS)
            or isinstance(planner, pomdp_py.PORPI)
            or isinstance(planner, pomdp_py.POUCT)):
            logging.info(f"{problem.agent.tree.children_data()}")
        logging.info(f"-------------------")
        logging.info(f"Belief:              {hist}")
        logging.info(f"-------------------")
        hist = {jsonpickle.encode(s) : p for s, p in hist.items()}
        logging.info(f"Belief (json):       {jsonpickle.encode(hist)}")
        logging.info(f"-------------------")
        logging.info(f"True State:          {problem.env.state}")
        logging.info(f"-------------------")
        logging.info(f"True State (json):   {jsonpickle.encode(problem.env.state)}")
        logging.info(f"-------------------")
        logging.info(f"Action:              {action}")

        """
        Step 2. Environment state transitions according to transition 
        model and agent receives reward and observation.
        Accounts for macro actions and observations.
        """

        # next_state, observation, reward, nsteps = sample_explict_models(T=problem.env.transition_model,
        #                                             O=problem.agent.observation_model,
        #                                             R=problem.env.reward_model,
        #                                             state=problem.env.state,
        #                                             action=action,
        #                                             discount_factor=planner.discount_factor)

        next_state, observation, reward, nsteps = problem.agent.blackbox_model.sample(state=problem.env.state,
                                                                                      action=action,
                                                                                      discount_factor=planner.discount_factor)

        problem.env.apply_transition(next_state)
        logging.info(f"-------------------")
        logging.info(f"Next State:          {next_state}")
        logging.info(f"-------------------")
        logging.info(f"Next State (json):   {jsonpickle.encode(next_state)}")
        logging.info(f"-------------------")
        logging.info(f"Observation:         {observation}")
        logging.info(f"Total Steps:         {total_steps}")
        logging.info(f"Reward:              {reward}")

        """
        Step 3. Update the history and belief.
        """
        problem.agent.update_history(action, observation)
        start_t = time.time()
        planner.update(problem.agent, action, observation)
        print(f'planner.update took {time.time() - start_t} seconds.')

        if kwargs.get("visualize_world"):
            problem.visualize_world()

        """
        Step 4. Record run statistics.
        """
        total_reward += reward
        total_reward_discounted += reward * (planner.discount_factor ** i)
        total_steps += nsteps

        logging.info(f"Cum. Disc. Reward:   {total_reward_discounted}")
        logging.info(f"Primitive Steps:     {nsteps}")

        if not isinstance(planner, pomdp_py.RefPol):
            logging.info(f"-------------------")
            logging.info(f"Planner (Num Sims):  {planner.last_num_sims}")
            logging.info(f"Planner (Time):      {planner.last_planning_time}")

        logging.info(f"=== STEP COMPLETE ===")
        i += 1

        """
        Step 5. If terminal 
        """
        # TODO: Logic here requires terminal and goal to be defined.
        if next_state.terminal:
            logging.info("Reached a terminal state")
            if next_state.is_goal:
                success = 1
            break



    mean_planning_time = np.mean(planning_times_per_step)
    mean_num_sims = np.mean(num_sims_per_step)

    return total_reward, total_reward_discounted, total_steps, success, mean_planning_time, mean_num_sims
