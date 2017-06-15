from scipy.stats import multivariate_normal as mvn
from numpy.random import multivariate_normal as mvnn
from scipy import random, linalg
from scipy.stats import rv_continuous
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import community
import itertools

from spatial_network import PlaneNode, SpatialNetwork


def random_nodelist(num_points=20, num_communities=5):
    centers = [[np.real(0.7 * np.exp(1j * 2 * np.pi * k / num_communities)),
                np.imag(0.7 * np.exp(1j * 2 * np.pi * k / num_communities))]
               for k in range(num_communities)]

    all_pts = [mvnn(c, 0.005 * np.eye(2), 10).tolist() for c in centers]
    pts = np.array(list(itertools.chain.from_iterable(all_pts)))
    nodelist = [PlaneNode(pts[i, :], label=str(i)) for i in range(pts.shape[0])]

    return nodelist


def uniform_nodelist(num_points=50):
    pts = 1 - 2 * np.random.random((num_points, 2))
    nodelist = [PlaneNode(pts[i, :], label=str(i)) for i in range(pts.shape[0])]

    return nodelist

def hub_spoke_nodelist(num_points=10, num_communities=5):
    centers = [[np.real(0.6 * np.exp(1j * 2 * np.pi * k / num_communities)),
                np.imag(0.6 * np.exp(1j * 2 * np.pi * k / num_communities))]
               for k in range(num_communities)]

    print np.array(centers)

    all_pts = [[[c1 + 0.1 * np.cos(2 * np.pi * k / num_points), c2 + 0.1 * np.sin(2 * np.pi * k / num_points)]
               for k in range(num_points)] + [[c1, c2]] for [c1, c2] in centers]
    pts = np.array(list(itertools.chain.from_iterable(all_pts)))
    nodelist = [PlaneNode(pts[i, :], label=str(i)) for i in range(pts.shape[0])]

    return nodelist



if __name__ == "__main__":
    np.set_printoptions(precision=1)


    def get_plots():
        pts = np.array([[0.0, 1.0],
                        [-.25, .5],
                        [-.25,-.5],
                        [0.0,  -1],
                        [1.0, 0.0]])

        edge_mat = np.array([[0, 1, 0, 0, 1],
                             [1, 0, 1, 0, 0],
                             [0, 1, 0, 1, 0],
                             [0, 0, 1, 0, 1],
                             [1, 0, 0, 1, 0]])

        nodelist = [PlaneNode(pts[i, :], label=str(i)) for i in range(pts.shape[0])]
        network = SpatialNetwork(nodelist)
        network.update_graph(edge_mat)
        network.draw_network(node_size=30)

        plt.axis('off')
        plt.savefig('delta_ex_highdelta.png', dpi=800)


    def plot_random_map():
        #nodelist = hub_spoke_nodelist()
        #nodelist = random_nodelist()
        nodelist = uniform_nodelist()

        d_mat = np.array([[np.linalg.norm(n1.pos - n2.pos) for n1 in nodelist] for n2 in nodelist])
        d_mat += np.eye(len(nodelist))
        w_mat = np.array(1.0 / d_mat - np.eye(len(nodelist)), dtype=[('weight', float)])
        network = nx.from_numpy_matrix(w_mat)
        partition = community.best_partition(network)

        pos = {idx: n.pos for (idx, n) in enumerate(nodelist)}
        size = float(len(set(partition.values())))
        cmap = plt.get_cmap('jet')
        network = SpatialNetwork(nodelist)
        network.draw_network()
        plt.axis('off')
        plt.savefig('no_delta_network.png', dpi=800)
        #plt.show()
        plt.close()

        count = 0
        for com in set(partition.values()):
            count = count + 1.
            list_nodes = [nodes for nodes in partition.keys()
                          if partition[nodes] == com]
            nx.draw_networkx_nodes(network, pos, list_nodes, node_size=20,
                                   node_color=cmap(count / size))

        network.nearest_neighbors_initialization()
        network.delta = 0.0
        network.optimize_network(reinit=False, verbose=True, num_trials=5)
        network.draw_network()
        plt.axis('off')
        plt.savefig('low_delta_network.png', dpi=800)
        #plt.show()
        plt.close()

        count = 0
        for com in set(partition.values()):
            count = count + 1.
            list_nodes = [nodes for nodes in partition.keys()
                          if partition[nodes] == com]
            nx.draw_networkx_nodes(network, pos, list_nodes, node_size=20,
                                   node_color=cmap(count / size))

        network.hub_initialization(hub_connections=2)
        network.delta = 0.9
        network.draw_network()
        plt.axis('off')
        plt.savefig('high_delta_network_preoptimize.png', dpi=800)
        #plt.show()
        plt.close()

        count = 0
        for com in set(partition.values()):
            count = count + 1.
            list_nodes = [nodes for nodes in partition.keys()
                          if partition[nodes] == com]
            nx.draw_networkx_nodes(network, pos, list_nodes, node_size=20,
                                   node_color=cmap(count / size))


        network.hub_initialization(hub_connections=3, random_edges=True)
        network.optimize_network(reinit=False, verbose=True, num_trials=5)
        network.draw_network()
        plt.axis('off')
        plt.savefig('high_delta_network.png', dpi=800)
        #plt.show()
        plt.close()




    #get_plots()
    plot_random_map()