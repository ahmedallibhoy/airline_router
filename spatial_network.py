import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from random import random
import operator
import community


# Community detection


def get_edge_mat(edge_list, nnodes):
    edge_mat = np.zeros((nnodes, nnodes))
    start = 0
    for idx in range(nnodes):
        edge_mat[idx, idx + 1:] = edge_list[start:(nnodes - idx - 1 + start)]
        start += nnodes - idx - 1
    return edge_mat + edge_mat.T

class PlaneNode(object):

    def __init__(self, pos, weight=0.5, label=None, draw_coord=None):
        self.pos = np.array(pos)

        if draw_coord is None:
            self.draw_coord = self.pos
        else:
            self.draw_coord = np.array(draw_coord)

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

class SpatialNetwork(object):

    def __init__(self, nodelist, gamma=0.1, delta=0.0):
        self.nodelist = nodelist
        self.num_nodes = len(nodelist)
        self.distance_matrix = np.array([[np.linalg.norm(n1.pos - n2.pos) for n1 in nodelist] for n2 in nodelist])
        self.weight_matrix = np.array([[n1.weight + n2.weight for n1 in nodelist] for n2 in nodelist])

        self.inv_distance_matrix = 1.0 / (self.distance_matrix + np.eye(len(nodelist))) - np.eye(len(nodelist))
        self.graph = nx.Graph()

        for (idx, node) in enumerate(self.nodelist):
            self.graph.add_node(idx)

        self.gamma = gamma
        self.delta = delta
        self.initialized = False

    def hub_initialization(self, hub_connections=4, random_edges=False):
        w_network = nx.Graph(self.inv_distance_matrix)
        network = nx.from_numpy_matrix(np.array(self.distance_matrix, dtype=[('length', float)]))

        partition = community.best_partition(w_network)
        num_communities = max(partition.values()) + 1
        communities = [list() for i in range(num_communities)]

        for (node, comm) in partition.iteritems():
            communities[comm] += [node]

        sub_edge_mat = [np.zeros((self.num_nodes, self.num_nodes)) for comm in communities]
        edge_mat = np.zeros((self.num_nodes, self.num_nodes))

        #edge_mat = np.ones((self.num_nodes, self.num_nodes))
        for idx, comm in enumerate(communities):
            for i in range(len(comm)):
                for j in range(i, len(comm)):
                    sub_edge_mat[idx][comm[i], comm[j]] = 1.0
                    sub_edge_mat[idx][comm[j], comm[i]] = 1.0

        for idx, mat in enumerate(sub_edge_mat):
            sub_edge_mat[idx] = np.array(np.multiply(self.distance_matrix, mat), dtype=[('length', float)])

        subgraphs = [nx.from_numpy_matrix(A) for A in sub_edge_mat]

        for idx, comm in enumerate(communities):
            sub_mat = np.zeros((self.num_nodes, self.num_nodes))

            centrality = nx.closeness_centrality(subgraphs[idx], distance='length')
            rank = [(node, rank) for (node, rank) in centrality.iteritems() if node in communities[idx]]
            rank = sorted(rank, key=lambda x: x[1])

            offset = np.array([1 if i in comm else 0 for i in range(self.num_nodes)]).reshape((self.num_nodes, 1))
            edge_mat[:, rank[-1][0]] = edge_mat[:, rank[-1][0]] + offset.T
            edge_mat[rank[-1][0], :] = edge_mat[rank[-1][0], :] + offset.T

        for i in range(len(communities)):
            for j in range(len(communities)):
                centrality_i = nx.closeness_centrality(subgraphs[i], distance='length')
                centrality_j = nx.closeness_centrality(subgraphs[j], distance='length')

                i_rank = [(node, rank) for (node, rank) in centrality_i.iteritems() if node in communities[i]]
                j_rank = [(node, rank) for (node, rank) in centrality_j.iteritems() if node in communities[j]]
                i_rank = sorted(i_rank, key=lambda x: x[1])
                j_rank = sorted(j_rank, key=lambda x: x[1])

                comm_cons = [i_rank[-m][0] for m in range(1, min(hub_connections, len(i_rank)))] + \
                            [j_rank[-n][0] for n in range(1, min(hub_connections, len(j_rank)))]

                for k in range(len(comm_cons)):
                    for l in range(len(comm_cons)):
                        edge_mat[comm_cons[k], comm_cons[l]] = 1.0
                        edge_mat[comm_cons[l], comm_cons[k]] = 1.0

        if random_edges == True:
            for i in range(self.num_nodes):
                distances = sorted(list(enumerate(self.distance_matrix[i, :])), key=lambda x: x[1])

                for j in distances[:2]:
                    edge_mat[i, j[0]] = 1
                    edge_mat[j[0], i] = 1

        for i in range(self.num_nodes):
            edge_mat[i, i] = 0

        self.update_graph(edge_mat)
        self.initialized = True

    def disconnected_hub_initialization(self):
        w_network = nx.Graph(self.inv_distance_matrix)
        network = nx.from_numpy_matrix(np.array(self.distance_matrix, dtype=[('length', float)]))

        partition = community.best_partition(w_network)
        num_communities = max(partition.values()) + 1
        communities = [list() for i in range(num_communities)]

        for (node, comm) in partition.iteritems():
            communities[comm] += [node]

        sub_edge_mat = [np.zeros((self.num_nodes, self.num_nodes)) for comm in communities]
        edge_mat = np.zeros((self.num_nodes, self.num_nodes))

        # edge_mat = np.ones((self.num_nodes, self.num_nodes))
        for idx, comm in enumerate(communities):
            for i in range(len(comm)):
                for j in range(i, len(comm)):
                    sub_edge_mat[idx][comm[i], comm[j]] = 1.0
                    sub_edge_mat[idx][comm[j], comm[i]] = 1.0

        for idx, mat in enumerate(sub_edge_mat):
            sub_edge_mat[idx] = np.array(np.multiply(self.distance_matrix, mat), dtype=[('length', float)])

        subgraphs = [nx.from_numpy_matrix(A) for A in sub_edge_mat]

        for idx, comm in enumerate(communities):
            sub_mat = np.zeros((self.num_nodes, self.num_nodes))

            centrality = nx.closeness_centrality(subgraphs[idx], distance='length')
            rank = [(node, rank) for (node, rank) in centrality.iteritems() if node in communities[idx]]
            rank = sorted(rank, key=lambda x: x[1])

            offset = np.array([1 if i in comm else 0 for i in range(self.num_nodes)]).reshape((self.num_nodes, 1))
            edge_mat[:, rank[-1][0]] = edge_mat[:, rank[-1][0]] + offset.T
            edge_mat[rank[-1][0], :] = edge_mat[rank[-1][0], :] + offset.T

        for i in range(self.num_nodes):
            edge_mat[i, i] = 0

        self.update_graph(edge_mat)
        self.initialized = True

    def initialize_graph(self):
        w_network = nx.Graph(self.inv_distance_matrix)
        network = nx.from_numpy_matrix(np.array(self.distance_matrix, dtype=[('length', float)]))

        partition = community.best_partition(w_network)
        num_communities = max(partition.values()) + 1
        communities = [list() for i in range(num_communities)]

        for (node, comm) in partition.iteritems():
            communities[comm] += [node]

        sub_edge_mat = [np.zeros((self.num_nodes, self.num_nodes)) for comm in communities]
        edge_mat = np.zeros((self.num_nodes, self.num_nodes))

        #edge_mat = np.ones((self.num_nodes, self.num_nodes))
        for idx, comm in enumerate(communities):
            for i in range(len(comm)):
                for j in range(i, len(comm)):
                    sub_edge_mat[idx][comm[i], comm[j]] = 1.0
                    sub_edge_mat[idx][comm[j], comm[i]] = 1.0
                    edge_mat[comm[j], comm[i]] = 1.0
                    edge_mat[comm[i], comm[j]] = 1.0

        for idx, mat in enumerate(sub_edge_mat):
            sub_edge_mat[idx] = np.array(np.multiply(self.distance_matrix, mat), dtype=[('length', float)])

        for i in range(len(communities)):
            for j in range(len(communities)):
                centrality_i = nx.closeness_centrality(network, distance='length')
                centrality_j = nx.closeness_centrality(network, distance='length')

                i_rank = [(node, rank) for (node, rank) in centrality_i.iteritems() if node in communities[i]]
                j_rank = [(node, rank) for (node, rank) in centrality_j.iteritems() if node in communities[j]]
                i_rank = sorted(i_rank, key=lambda x: x[1])
                j_rank = sorted(j_rank, key=lambda x: x[1])

                comm_cons = [i_rank[-m][0] for m in range(1, min(3, len(i_rank)))] + \
                            [j_rank[-n][0] for n in range(1, min(3, len(j_rank)))]

                for k in range(len(comm_cons)):
                    for l in range(len(comm_cons)):
                        edge_mat[comm_cons[k], comm_cons[l]] = 1.0
                        edge_mat[comm_cons[l], comm_cons[k]] = 1.0

        for i in range(self.num_nodes):
            edge_mat[i, i] = 0

        self.update_graph(edge_mat)
        self.initialized = True

    def nearest_neighbors_initialization(self, nearest_neighbors=8):
        edge_mat = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            distances = sorted(list(enumerate(self.distance_matrix[i, :])), key=lambda x: x[1])

            for j in distances[:nearest_neighbors]:
                edge_mat[i, j[0]] = 1
                edge_mat[j[0], i] = 1

        edge_mat = edge_mat - np.eye(self.num_nodes)
        self.update_graph(edge_mat)
        self.initialized = True


    def update_graph(self, edge_mat):
        A =  np.multiply(self.distance_matrix, edge_mat)
        route_mat = np.array(A, dtype=[('length', float)])
        self.graph = nx.from_numpy_matrix(route_mat)

    def get_graph_cost(self, graph):
        len_matrix = nx.floyd_warshall_numpy(graph, weight='length')
        indices = np.where(len_matrix == float('inf'))
        len_matrix[indices[0], indices[1]] = 2e10

        edge_mat = nx.to_numpy_matrix(graph)
        effective_len_matrix = (1 - self.delta) * len_matrix + self.delta

        total_path_length = np.sum(np.multiply(len_matrix, edge_mat))
        maintenance_cost = np.sum(np.multiply(self.weight_matrix, effective_len_matrix))

        return (1 - self.gamma) * total_path_length + self.gamma * maintenance_cost

    def get_cost(self, edge_mat=None):
        if edge_mat is None:
            return self.get_graph_cost(self.graph)

        A = np.multiply(self.distance_matrix, edge_mat)
        route_mat = np.array(A, dtype=[('length', float)])
        graph = nx.from_numpy_matrix(route_mat)

        return self.get_graph_cost(graph)

    def try_remove_edges(self, prob_remove):
        cost = self.get_cost()
        removed = False

        ebc = nx.edge_betweenness_centrality(self.graph, weight='length')
        max_ebc = max(ebc.values())

        for m in range(self.num_nodes * (self.num_nodes - 1)):
            edge_mat = nx.to_numpy_matrix(self.graph)

            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    if (i, j) in ebc:
                        p_offset = 1 - ebc[(i, j)] / max_ebc
                    elif (j, i) in ebc:
                        p_offset = 1 - ebc[(j, i)] / max_ebc
                    else:
                        p_offset = 0

                    if random() < prob_remove + 0.2 * p_offset:
                        removed = True
                        edge_mat[i, j] = 0
                        edge_mat[j, i] = 0

            new_cost = self.get_cost(edge_mat)

            if new_cost < cost:
                self.update_graph(edge_mat)
                return False, removed, edge_mat, new_cost

        return True, removed, nx.to_numpy_matrix(self.graph), cost

    def optimize_network(self, prob_remove=0.05, reinit=True, num_trials=1, verbose=False):
        if reinit or (not self.initialized):
            self.initialize_graph()

        edge_mat_list = [None] * (num_trials + 1)
        edge_mat = nx.to_numpy_matrix(self.graph)
        cost = self.get_cost()

        edge_mat_list[0] = (edge_mat, cost)

        for i in range(num_trials):
            if verbose:
                print "Beginning trial %d..." % (i + 1)

            finished = False
            removed = False
            while not (finished and removed):
                finished, removed, edge_mat, cost = self.try_remove_edges(prob_remove)
            edge_mat_list[i + 1] = (edge_mat, cost)

            if verbose:
                print "Finished, with cost=%.2f" % cost
            self.update_graph(edge_mat_list[0][0])

        edge_mat_list = sorted(edge_mat_list, key=lambda x: x[1])
        self.update_graph(edge_mat_list[0][0])
        return edge_mat_list[0][1]

    def get_edge_list(self):
        return self.graph.edges()

    def draw_network(self, node_size=10):
        #plt.figure()
        pos = {idx: node.draw_coord for idx, node in enumerate(self.nodelist)}
        print self.graph.edges()
        nx.draw_networkx_nodes(self.graph, pos, node_size=node_size, node_color='k')
        nx.draw_networkx_edges(self.graph, pos, alpha=0.9, edge_color='k')
        #nx.draw_networkx_edges(self.graph, pos, alpha=0.9, edge_color=['b', 'k', 'b', 'b', 'k'])

        #plt.show()


if __name__ == "__main__":
    pass