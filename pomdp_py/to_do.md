## To Do list

# High Priority
- [ ] Double-check SIR filter.
- [ ] Double-check run accounting.
- [ ] Add holonomic environments.
- [ ] Implement ABT.
- [ ] Add Dubins' helicopter model.
- [ ] Implement environment with changing victim locations.
- [ ] Expand functions relating to objectives to allow for observations too (e.g. capture problem)

# Low Priority
- [ ] Rename PORPI to PORPP
- [ ] Change shape of belief in 3D environment.
- [ ] Add parameter logging functionality.
- [ ] Add meshes and heightfields to FCL environments.
- [ ] Run POMCP and test belief update for macro actions.
- [ ] Can the path planning module be cythonized?
- [ ] Set up additional environments.
- [ ] Check planning time is correct.
- [ ] Fix TQDM frac issue
- [ ] Edit logger to remove INFO: tags.
- [ ] Generalise logger for multiple environments.
- [ ] Add a ceiling to the FCLMaze environment.
- [ ] Build loadable PRM from file.
- [ ] In run_experiments randomise the initial state based on initial belief.
- [ ] Debug FCL environment.
- [ ] Validate that we can achieve must faster update and planning time using FCL.
- [ ] Create version of PORPI that avoids mellowmax.

# Open Questions
- [ ] Can we avoid the tree structure altogether in our planner? After all, isn't the advantage of our approach the fact that we can just evaluate averages?
- [ ] reward average or reward?
- [ ] Can we scale the reward into the range [0,1] for the computing the logsumexp?

# Archive
- [x] Update choice of waypoint.
- [x] Add an exploitation parameter to PORPI.
- [x] Double-check and diagnose run sir resampling issue.
- [x] Ensure json compatibility for log-files from server.
- [x] Try uniformly sampling key configs for Corsica problems (doesn't work well...)
- [x] Write no planning planner (RefPol).
- [x] Check POMCP implementation.
- [x] Add __str__ with parameters to each planner + to test_runner.
- [x] Redocument RefSolverNIPS and RefSolver.
- [x] Clean up parameters for PODPP and rename?? Renamed to PORPI
- [x] Add RefSolverNIPS (check if using eta param).
- [x] Reset columns in Corsica danger zones.
- [X] Add stochasticity to stress model transition.
- [x] Backtest compatibility with other environments.
- [x] Generalize replay functionality.
- [x] Add a parser for run scripts.
- [x] Debug collision checking for environment evolution for Corsica.
- [x] Update test runner to generalize across all planners (add json.pickle)
- [x] Repair generality of solver logger and visualiser.
- [x] Review reference policies for respective environments.
- [x] Add RefSolver (clean up code).
- [x] Improve POMCP to perform SIR resampling.
- [x] Add POMCP for benchmarking.
- [x] Remove definition of MacroObservation equivalence from framework.
- [x] Add replay logger.
- [x] Add belief visualizer in motion planning problems.
- [x] Add SIR sampler to planner.
- [x] Cythonize plugins.
- [x] Validate cythonized plugins and refactor.
- [x] Add log files to gitignore.
- [x] Improve speed of SIR sampler.
- [x] Add belief visualiser to the logs.
- [x] Validate the updated SIR sampler.
- [x] Update the observation model for landmarks to use Gaussian noise and exaggerate sampling towards heavily weighted state particles.
- [x] Implement other PRM settings (i.e. sample points heavily around milestones).
- [x] Remove pure python Nav2DContinuous plugins.
- [x] Use dictionaries instead of lists in test_outputs and check run indexing + avg num_sims and planning_time.
- [x] Add PRM nodes and edges to the log file.
- [x] Incorporate fast collision checking in environments.
- [x] Proof of convergence with covering number of reachable belief space as complexity input.
- [x] Use offline heuristic for rollout.
- [X] Resample state from belief after action sample to ensure beliefs are properly maintained.