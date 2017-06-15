import numpy as np
import operator
from mpl_toolkits.basemap import *
from spatial_network import PlaneNode, SpatialNetwork
import matplotlib.pyplot as plt
import community
import networkx as nx

def parse_airports():
    airports = dict()

    with open('data/airports.dat') as f:
        for line in f:
            data = line.split(',')
            country = data[3].replace("\"", "")
            if country == "United States" or country == "USA" or country == "United States of America":
                airports[int(data[0])] = [data[1].replace("\"", ""), # Name
                    data[2].replace("\"", ""), # City
                    data[4].replace("\"", ""), # IATA
                    data[5].replace("\"", ""), # ICAO
                    float(data[6]),
                    float(data[7]), 0]
    return airports

def parse_routes(pre_data):
    new_data = pre_data

    with open('data/routes.dat') as f:
        for line in f:
            data = line.split(',')

            if data[3] != '\N' and int(data[3]) in pre_data:
                new_data[int(data[3])][6] = new_data[int(data[3])][6] + 1
            if data[5] != '\N' and int(data[5]) in pre_data:
                new_data[int(data[5])][6] = new_data[int(data[5])][6] + 1

    return new_data


if __name__ == "__main__":
    data = parse_airports()
    data = parse_routes(data)

    sorted_data = sorted(data.items(), key=lambda x: -x[1][6])[:50]
    airports = np.array([[s[1][4], s[1][5]] for s in sorted_data])

    m = Basemap(projection='lcc', llcrnrlon=-122.63, llcrnrlat=22.57901, urcrnrlon=-60.63, urcrnrlat = 46.24,
                lat_0 = 40, lon_0 = -100, resolution = 'l')
    #m.drawstates()
    m.drawcountries()
    #m.fillcontinents()
    m.drawcoastlines()

    def normalize_x(x):
        return 2 * x / m.xmax - 1

    def normalize_y(y):
        return 2 * y / m.ymax - 1

    def proj_x(nx):
        return m.xmax / 2 * (nx + 1)

    def proj_y(ny):
        return m.ymax / 2 * (ny + 1)

    x,y = m(airports[:, 1], airports[:, 0])

    plane_coordinates = np.array([x, y]).T
    normalized_coordinates = np.array([normalize_x(plane_coordinates[:, 0]),
                                       normalize_y(plane_coordinates[:, 1])]).T

    nodelist = [PlaneNode(n, draw_coord=d) for (n, d) in zip(normalized_coordinates.tolist(),
                plane_coordinates.tolist()) if -1 < n[0] < 1 and -1 < n[1] < 1]

    plt.scatter([n.draw_coord[0] for n in nodelist], [n.draw_coord[1] for n in nodelist], s=10, c='k')
    network = SpatialNetwork(nodelist)
    network.draw_network()
    m.drawcountries()
    m.fillcontinents(alpha=0.5)
    m.drawcoastlines()
    plt.axis('off')
    plt.savefig('us_no_edges.png', dpi=800)
    plt.close()


    d_mat = np.array([[np.linalg.norm(n1.draw_coord - n2.draw_coord) for n1 in nodelist] for n2 in nodelist])
    d_mat += np.eye(len(nodelist))
    w_mat = np.array(1.0 / d_mat - np.eye(len(nodelist)), dtype=[('weight', float)])
    n = nx.from_numpy_matrix(w_mat)
    partition = community.best_partition(n)

    pos = {idx: n.draw_coord for (idx, n) in enumerate(nodelist)}
    size = float(len(set(partition.values())))
    count = 0

    cmap = plt.get_cmap('jet')

    for com in set(partition.values()):
        count = count + 1.
        list_nodes = [nodes for nodes in partition.keys()
                      if partition[nodes] == com]
        nx.draw_networkx_nodes(n, pos, list_nodes, node_size=20,
                               node_color=cmap(count / size))

    network.hub_initialization(hub_connections=2)
    print network.optimize_network(reinit=False, verbose=True, num_trials=1)

    #print network.get_cost()
    #network.initialize_graph()
    network.draw_network()
    m.drawcountries()
    m.fillcontinents(alpha=0.5)
    m.drawcoastlines()
    plt.axis('off')
    plt.savefig('us_edges.png', dpi=800)
    plt.close()