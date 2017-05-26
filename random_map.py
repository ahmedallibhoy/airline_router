from scipy.stats import multivariate_normal as mvn
from numpy.random import multivariate_normal
from scipy import random, linalg
from scipy.stats import rv_continuous
from random import randint
import numpy as np
import matplotlib.pyplot as plt


def random_pdf(offset=None, rscale=.1, sscale=.001):
    if offset is None:
        offset = np.array([0, 0])

    m_angle = 2 * np.pi * random.random(1)[0]
    m_rad = rscale * random.random(1)[0]

    m = [m_rad * np.cos(m_angle), m_rad * np.sin(m_angle)] + offset

    for i in [0, 1]:
        if m[i] < -1:
            m[i] = -1
        if m[i] > 1:
            m[i] = 1

    C = sscale * np.eye(2)
    rv = mvn(m, C)

    return rv, [m_rad * np.cos(m_angle), m_rad * np.sin(m_angle)] + offset


def random_map(num_metro=None):
    if num_metro is None:
        num_metro = randint(5, 15)

    metro_centers = [1 - 2 * random.random(2) for i in range(num_metro)]
    rpdf= [random_pdf(mloc, sscale=.01, rscale=0.3) for i in range(randint(50, 100)) for mloc in metro_centers]

    metros = [r[0] for r in rpdf]

    pts = np.array([r[1] for r in rpdf])

    C = 1.0 / len(metros)

    def sample(s):
        return mvn(metro_centers[random.randint(0, len(metro_centers))], .01 * np.eye(2)).rvs(s)

    return lambda x: C * sum([mvn.pdf(x) for mvn in metros]), sample, pts


class MapVar(rv_continuous):

    def __init__(self, pdf=None, *args):
        if pdf is not None:
            self.m_pdf = pdf
        else:
            self.m_pdf = random_map()

        rv_continuous.__init__(self, *args)

    def _pdf(self, x, *args):
        return self.m_pdf(x)


if __name__ == "__main__":
    def plot_random_map():
        x, y = np.mgrid[-1:1:.01, -1:1:.01]
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x;
        pos[:, :, 1] = y

        rmap, rsample, rpts = random_map()

        z = rmap(pos)
        plt.contourf(x, y, z)
        plt.colorbar()

        plt.scatter(rpts[:, 0], rpts[:, 1], s=.1, c='k')

        s = rsample(1000)
        print s
        #plt.scatter([ss[0] for ss in s], [ss[1] for ss in s], c='k', s=10)
        plt.show()



    plot_random_map()