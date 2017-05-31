import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import basinhopping


def shortest_path(G, i, j):
    try:
        return nx.shortest_path_length(G, i, j)
    except nx.NetworkXNoPath:
        return float('inf')


class PlaneNode(object):

    def __init__(self, pos, weight=0.5, label=None):
        self.pos = np.array(pos)
        self.label = label
        self.weight = weight


class SpatialNetwork(object):

    def __init__(self, nodelist, gamma=1e-14, delta=0.5):
        self.nodelist = nodelist
        self.num_nodes = len(nodelist)
        self.distance_matrix = np.array([[np.linalg.norm(n1.pos - n2.pos) for n1 in nodelist] for n2 in nodelist])
        self.weight_matrix = np.array([[n1.weight + n2.weight for n1 in nodelist] for n2 in nodelist])
        self.graph = nx.Graph()

        for (idx, node) in enumerate(self.nodelist):
            self.graph.add_node(idx)

        self.gamma = gamma
        self.delta = delta

    def get_cost(self, edge_mat):
        route_mat = np.array(self.distance_matrix * edge_mat, dtype=[('length', float)])
        self.graph = nx.from_numpy_matrix(route_mat)

        path_lengths = nx.shortest_path_length(self.graph, weight='length')
        len_matrix = np.zeros((self.num_nodes, self.num_nodes))

        len_matrix = np.array([[shortest_path(self.graph, i, j) for i in range(self.num_nodes)]
                               for j in range(self.num_nodes)])

        #for (from_node, nodelist) in path_lengths.iteritems():
        #    for to_node in range(self.num_nodes):
        #        if to_node not in nodelist:
        #            len_matrix[from_node, to_node] = float('inf') #
        #            return float('inf') # Automatically return inf for unconnected network (Gastner-Newman)
        #        else:
        #            len_matrix[from_node, to_node] = nodelist[to_node]

        effective_len_matrix = (1 - self.delta) * len_matrix + self.delta

        # Scaled down by factor of 1/2 to account for double counting
        total_path_length = 0.5 * np.sum(len_matrix * np.array(nx.to_numpy_matrix(self.graph)))
        maintenance_cost = 0.5 * np.sum(self.weight_matrix * effective_len_matrix)

        return total_path_length + self.gamma * maintenance_cost

    def optimize_network(self):
        def objective_function(edge_mat, instance):
            return instance.get_cost(1.0 * (edge_mat.reshape(self.num_nodes, self.num_nodes) > 0))
            # Might be equivalent...
            # return instance.get_cost(1.0 * (edge_mat.reshape(self.num_nodes, self.num_nodes) != 0))

        # Simulated Annealing is deprecated in SciPy
        result = basinhopping(objective_function,
                              np.array([1.0  if i % self.num_nodes -1 != 0 else 0.0
                                        for i in range(self.num_nodes * self.num_nodes)]),
                              minimizer_kwargs={'args': (self)}, disp=True)

        edge_mat = 1.0 * (result.x.reshape(self.num_nodes, self.num_nodes) > 0)
        route_mat = np.array(self.distance_matrix * edge_mat, dtype=[('length', float)])
        self.graph = nx.from_numpy_matrix(route_mat)



if __name__ == "__main__":
    pass