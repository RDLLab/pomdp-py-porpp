"""This module facilitates construction and analysis of Expansive-Spaces Trees in pybullet environments."""

# TODO: This code is a work-in-progress.

import time
import networkx as nx
from pomdp_py.problems.motion_planning.environments.pyb_env import PyBulletEnv

class EST():

    def __init__(self,
                 pyb_env: PyBulletEnv,
                 count_radius: int,
                 start_config,
                 goal_config,
                 weight="degree"
                 ):

        """
        pyb_env (PyBulletEnv): The PyBullet environment.
        """

        self.pyb_env = pyb_env

        """
        Set up graph and KD tree.
        """

        self.est = self.initialize_est(start_config)

        start_t = time.time()
        self.est, self.kd_tree = self.construct_est()
        print(f'PRM initialization time: {time.time() - start_t}')

        start_t = time.time()
        self.all_pairs_shortest_paths = dict(nx.all_pairs_dijkstra(self.est))
        print(f'All pairs shortest paths computation time: {time.time() - start_t}')


    def initialize_est(self, start_config):
        g = nx.Graph()
        g.add_node(start_config)

    def extend_est(self):


    def sample_config(self):
