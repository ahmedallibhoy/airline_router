import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import basinhopping


def get_edge_mat(edge_list, nnodes):
    edge_mat = np.zeros((nnodes, nnodes))
    start = 0
    for idx in range(nnodes):
        edge_mat[idx, idx + 1:] = edge_list[start:(nnodes - idx - 1 + start)]
        start += nnodes - idx - 1
    return edge_mat + edge_mat.T

class PlaneNode(object):

    def __init__(self, pos, weight=0.5, label=None):
        self.pos = np.array(pos)
        self.label = label
        self.weight = weight

    def distance_to(self, other):
        return np.linalg.norm(self.pos - other.pos)


class GlobeNode(object):

    def __init__(self, lat, long, weight=0.5, label=None):
        self.lat =  lat
        self.long = long
        self.label = label
        self.weight = weight

    def distance_to(self, other):
        raise NotImplementedError


class HBounds(object):

    def __init__(self, nnodes, lower=0.0, upper=1.0):
        self.nnodes = nnodes
        self.lower = lower
        self.upper = upper

    def __call__(self, **kwargs):
        x = kwargs['x_new']

        # Construct Graph Laplacian
        edge_mat = get_edge_mat(1.0 * (x > 0), self.nnodes)
        ones = np.ones((self.nnodes))

        D = np.diag(np.dot(edge_mat, ones))
        L = D - edge_mat

        # Check multiplicity of eigenvalue corresponding to 0 == 1
        w = np.linalg.eigvals(L)
        s = np.where(np.abs(np.array(sorted(w))) < 1.0e-8)[0][-1]
        return s == 0

class SpatialNetwork(object):

    def __init__(self, nodelist, gamma=0, delta=0.5):
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
        len_matrix = nx.floyd_warshall_numpy(self.graph, weight='length')

        indices = np.where(len_matrix == float('inf'))
        len_matrix[indices[0], indices[1]] = 10000.0
        effective_len_matrix = (1 - self.delta) * len_matrix + self.delta

        # Scaled down by factor of 1/2 to account for double counting
        total_path_length = 0.5 * np.sum(len_matrix * np.array(nx.to_numpy_matrix(self.graph)))
        maintenance_cost = 0.5 * np.sum(self.weight_matrix * effective_len_matrix)

        return total_path_length + self.gamma * maintenance_cost

    def optimize_network(self):
        def objective_function(edge_list, instance):
            return instance.get_cost(1.0 * get_edge_mat(edge_list, instance.num_nodes) > 0.0)

        initial_guess = np.zeros(((self.num_nodes - 1) * (self.num_nodes - 2)))
        cumm_idx = 0

        for idx in range(self.num_nodes):
            initial_guess[cumm_idx] = 1.0
            cumm_idx += self.num_nodes - idx

        # Simulated Annealing is deprecated in SciPy
        result = basinhopping(objective_function, initial_guess, minimizer_kwargs={'args': (self)}, disp=True,
                              accept_test=HBounds(self.num_nodes), niter=500)

        edge_mat = get_edge_mat(result.x, self.num_nodes)

        print edge_mat
        route_mat = np.array(self.distance_matrix * edge_mat, dtype=[('length', float)])
        self.graph = nx.from_numpy_matrix(route_mat)

        print nx.adjacency_matrix(self.graph)


    def draw_network(self):
        #plt.figure()
        pos = {idx: node.pos for idx, node in enumerate(self.nodelist)}
        #nx.draw_networkx_nodes(self.graph, pos, node_size=1, node_color='k')
        nx.draw_networkx_edges(self.graph, pos, alpha=0.5, style='dotted')
        plt.show()


if __name__ == "__main__":
    pass