"""This module facilitates construction and analysis of Probabilistic Roadmaps in pybullet environments."""

import time
import numpy as np
import networkx as nx
from scipy.spatial import KDTree
import pybullet as pyb

from pomdp_py.problems.motion_planning.environments.pyb_env import PyBulletEnv

class PRM():

    def __init__(self,
                 pyb_env: PyBulletEnv,
                 dimension,
                 lower_bounds,
                 upper_bounds,
                 num_nodes=1000,
                 num_neighbours=5,
                 max_degree=10,
                 points_per_segment=30,
                 sampling_dist="Uniform",
                 gaussian_noise=1.,
                 pyb_dimension=6,
                 nodes_to_include=None,
                 assert_connected=False):

        """
        pyb_env (PyBulletEnv): The PyBullet environment.
        dimension (int): The dimension of the configuration space.
        lower_bounds (list): Lower limits in each dimension.
        upper_bounds (list): Upper limits in each dimension.
        num_nodes (int): Number of nodes required in the PRM.
        num_neighbours (int): Number of closest neighbours to examine for each configuration.
        max_degree (int): The maximum number of out edges from a node.
        points_per_segment (int): Number of evenly spaced points used to represent a line
            segment during collision checks.
        sampling_dist (str): The sampling distribution for new nodes. Defaults to "Uniform".
        gaussian_noise (float): The noise of the Gaussian sampler.
        pyb_dimension (int): The dimension required by the PyBullet environment.
        nodes_to_include (list): If given, the PRM will be constructed with these nodes included.
            If sampler="Gaussian", points will be sampled around these points with Gaussian noise.
        assert_connected (bool): Requires that the PRM be connected.
        """

        if len(lower_bounds) != dimension or len(upper_bounds) != dimension:
            raise ValueError("Bound lists not consistent with dimension.")

        self.pyb_env = pyb_env
        self.dimension = dimension
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.num_nodes = num_nodes
        self.num_neighbours = num_neighbours
        self.max_degree = max_degree
        self.points_per_segment = points_per_segment
        self.sampling_dist = sampling_dist
        self.gaussian_noise = gaussian_noise
        self.pyb_dimension = pyb_dimension
        self.nodes_to_include = nodes_to_include
        self.assert_connected = assert_connected

        """
        Set up graph and KD tree.
        """

        start_t = time.time()
        self.prm, self.kd_tree = self.construct_prm(self.num_neighbours)
        print(f'PRM initialization time: {time.time() - start_t}')

        start_t = time.time()
        self.all_pairs_shortest_paths = dict(nx.all_pairs_dijkstra(self.prm))
        print(f'All pairs shortest paths computation time: {time.time() - start_t}')

    """Add other sampling functions here"""
    # TODO: Incorporate a few different ones.
    def sampler(self):

        if self.sampling_dist == "Gaussian":
            if self.nodes_to_include is None or not len(self.nodes_to_include):
                raise Exception("No nodes provided. Please add nodes to nodes_to_include.")
            point = np.array(self.nodes_to_include[np.random.randint(len(self.nodes_to_include))])
            sample_point = point + np.random.normal(0, self.gaussian_noise, self.dimension)
            if all(sample_point >= self.lower_bounds) and all(sample_point <= self.upper_bounds):
                return np.concatenate((sample_point, np.zeros(self.pyb_dimension - self.dimension)))
            else:
                return np.concatenate(
                    (np.random.uniform(low=self.lower_bounds, high=self.upper_bounds, size=(1, self.dimension))[0],
                     np.zeros(self.pyb_dimension - self.dimension)))
        else:
            return np.concatenate((np.random.uniform(low=self.lower_bounds, high=self.upper_bounds, size=(1, self.dimension))[0],
                                   np.zeros(self.pyb_dimension - self.dimension)))

    def segment_invalid(self, p1, p2):

        """
        Checks whether the line segment between p1 and p2 can be connected without collisions.
        Returns True if it can't (i.e. there is a collision), False otherwise (i.e. no detected collisions).

        p1, p2 (np.array): Points.
        """

        dif = p2 - p1
        if np.linalg.norm(dif) < 0.0001:
            return self.pyb_env.collision_checker(p1)
        points = list(map(lambda i: p1 + dif * (i / self.points_per_segment), range(1, self.points_per_segment)))
        for p in points:
            if self.pyb_env.collision_checker(p):
                return True
        return False

    def construct_prm(self, num_neighbours=10, iterations=50):

        g = nx.Graph()

        i = 0
        while i < iterations:

            if self.nodes_to_include is not None:
                for p in self.nodes_to_include:
                    g.add_node(p + tuple(np.zeros(self.pyb_dimension - self.dimension)))

            num_nodes = g.number_of_nodes()

            # Sample required number of nodes.
            while num_nodes < self.num_nodes:
                p = self.sampler()
                if not self.pyb_env.collision_checker(p):
                    g.add_node(tuple(p))
                    num_nodes += 1

            kd_tree = KDTree(data=np.array(g.nodes))

            for p in g.nodes:
                distances, nearest_neighbours = kd_tree.query(x=p, k=min(num_neighbours+i, num_nodes-1))
                for i, q in enumerate(nearest_neighbours):
                    if not self.segment_invalid(np.array(p), kd_tree.data[q]):
                        g.add_edge(p, tuple(kd_tree.data[q]), weight=distances[i])
                        if g.degree(p) >= self.max_degree:
                            break

            if not self.assert_connected:
                return g, kd_tree

            if nx.is_connected(g):
                return g, kd_tree
            else:
                print(f"PRM is not connected. Incrementing neighbour candidates by 1 and trying again.")
            i += 1

        print(f"Connected PRM could not be constructed after {iterations} iterations.")

    def plot_prm(self, client_id):

        for e in self.prm.edges:
            # pyb.addUserDebugLine(lineFromXYZ=e[0][0:3], lineToXYZ=e[1][0:3], lineColorRGB=[0.6, 0.8, 0.9],
            #                      lineWidth=0.001, lifeTime=0, physicsClientId=client_id)
            pyb.addUserDebugLine(lineFromXYZ=e[0][0:3], lineToXYZ=e[1][0:3], lineColorRGB=[0, 0, 0],
                                 lineWidth=.1, lifeTime=0, physicsClientId=client_id)
            # Assumes that configurations are of the format (x, y, z, roll, pitch, yaw).
            time.sleep(60)

    def shortest_path_length(self, source, target):

        source_d, source_i = self.kd_tree.query(source)
        target_d, target_i = self.kd_tree.query(target)

        source_node = tuple(self.kd_tree.data[source_i])
        target_node = tuple(self.kd_tree.data[target_i])

        return nx.shortest_path_length(self.prm, source=source_node, target=target_node, weight='weight')

    def shortest_path_offline(self, source, target):
        """
        Returns the shortest path from source to target based on preprocessed all-pairs-shortest-path data.

        source (tuple): The source vertex co-ordinates.
        target (tuple): The target vertex co-ordinates.

        """

        source_d, source_i = self.kd_tree.query(source)
        target_d, target_i = self.kd_tree.query(target)

        source_node = tuple(self.kd_tree.data[source_i])
        target_node = tuple(self.kd_tree.data[target_i])

        if np.linalg.norm(np.array(source) - np.array(target)) < 1.5:
            return [target]

        # TODO: Check this.
        if source_node == target_node:
            return [target_node]

        try:
            return self.all_pairs_shortest_paths[source_node][1][target_node]
        except:
            if not nx.has_path(self.prm, source=source_node, target=target_node):
                print(f"No path between source: {source_node}, target: {target_node}")
            return []

    def shortest_path(self, source, target):
        """
        Returns the shortest path from source to target online.

        source (tuple): The source vertex co-ordinates.
        target (tuple): The target vertex co-ordinates.

        """
        source_d, source_i = self.kd_tree.query(source)
        target_d, target_i = self.kd_tree.query(target)

        source_node = tuple(self.kd_tree.data[source_i])
        target_node = tuple(self.kd_tree.data[target_i])

        if not nx.has_path(self.prm, source=source_node, target=target_node):
            raise ValueError('No path from source to target. Sample more points in PRM.')

        return nx.shortest_path(self.prm, source=source_node, target=target_node, weight='weight')

    def sample_out_edge(self, source):
        """
        Samples a random out edge from a given source node.
        """

        source_d, source_i = self.kd_tree.query(source)
        edges = self.prm.edges([source_i])
        edge_list = list(edges._viewer)
        real_edges = [e for e in edge_list if e[0] != e[1]]
        e = real_edges[np.random.randint(len(real_edges))]
        if e is None:
            raise ValueError("No out edge.")
        if e[0] == e[1]:
            raise ValueError(f"Edge {e} has zero length.")
        return e