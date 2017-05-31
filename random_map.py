from scipy.stats import multivariate_normal as mvn
from numpy.random import multivariate_normal
from scipy import random, linalg
from scipy.stats import rv_continuous
from random import randint
import numpy as np
import matplotlib.pyplot as plt

from spatial_network import PlaneNode, SpatialNetwork


def random_pdf(offset=None, rscale=.1, sscale=.001):
    if offset is None:
        offset = np.array([0, 0])

    m_angle = 2 * np.pi * random.random(1)[0]
    m_rad = rscale * random.random(1)[0]

    m = [m_rad * np.cos(m_angle), m_rad * np.sin(m_angle)] + offset
    C = sscale * np.eye(2)
    rv = mvn(m, C)

    return rv, m


def random_map(num_metro=None):
    if num_metro is None:
        num_metro = randint(1, 5)

    metro_centers = [1 - 2 * random.random(2) for i in range(num_metro)]
    rpdf= [random_pdf(mloc, sscale=.01, rscale=0.3) for i in range(randint(5, 7)) for mloc in metro_centers]
    metros = [r[0] for r in rpdf]

    rpts = [r[1] for r in rpdf if -1.0 < r[1][0] < 1.0 and -1 < r[1][1] < 1.0]
    upts = (1 - 2 * np.random.random((10, 2))).tolist()
    pts = np.array(rpts + upts)

    C = 1.0 / len(metros)

    nodelist = [PlaneNode(pts[i, :], label=str(i)) for i in range(pts.shape[0])]
    return lambda x: C * sum([mvn.pdf(x) for mvn in metros]), nodelist

if __name__ == "__main__":
    def plot_random_map():
        x, y = np.mgrid[-1:1:.01, -1:1:.01]
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y

        rmap, nodelist = random_map()

        z = rmap(pos)
        plt.contourf(x, y, z)
        plt.colorbar()

        plt.scatter([n.pos[0] for n in nodelist], [n.pos[1] for n in nodelist], s=.1, c='k')
        plt.show()

        network = SpatialNetwork(nodelist)
        print len(nodelist)

        edge_list = np.array([1.0  if i % network.num_nodes -1 != 0 else 0.0
                              for i in range(network.num_nodes * network.num_nodes)])
        edge_mat = edge_list.reshape((network.num_nodes, network.num_nodes))

        print network.get_cost(edge_mat)

        network.optimize_network()



    plot_random_map()