#!/usr/bin/python2
# encoding=utf-8

import collections
import datetime
import folium
import folium.plugins
import graph_drawing
import matplotlib.pyplot as plt
import networkx
import numpy as np
import os
import pandas
import sys


path_to_data = "data.csv"


class Graph:
    def __init__(self):
        self._graph = networkx.MultiDiGraph()
        # Coordinates are (Latitude, Longitude)
        self._resolution_degrees = (0.0006, 0.0006)
        # 1 World Trade Center NYC
        self._reference_point = (40.712748, -74.013379)

    def discretize_coordinates(self, coordinates):
        return (int((coordinates[0] - self._reference_point[0]) / self._resolution_degrees[0]),
                int((coordinates[1] - self._reference_point[1]) / self._resolution_degrees[1]))

    def node_to_coordinates(self, node_coordinates):
        return (node_coordinates[0] * self._resolution_degrees[0] + self._reference_point[0] + self._resolution_degrees[0] / 2.0,
                node_coordinates[1] * self._resolution_degrees[1] + self._reference_point[1] + self._resolution_degrees[1] / 2.0)

    def node_to_borders(self, node_coordinates):
        return [(node_coordinates[0] * self._resolution_degrees[0] + self._reference_point[0],
                node_coordinates[1] * self._resolution_degrees[1] + self._reference_point[1]),
                (node_coordinates[0] * self._resolution_degrees[0] + self._reference_point[0] + self._resolution_degrees[0],
                 node_coordinates[1] * self._resolution_degrees[1] + self._reference_point[1]),
                (node_coordinates[0] * self._resolution_degrees[0] + self._reference_point[0],
                node_coordinates[1] * self._resolution_degrees[1] + self._reference_point[1] + self._resolution_degrees[1]),
                (node_coordinates[0] * self._resolution_degrees[0] + self._reference_point[0] + self._resolution_degrees[0],
                node_coordinates[1] * self._resolution_degrees[1] + self._reference_point[1] + self._resolution_degrees[1])]

    def filter_edges(self, predicate, item):
        edges_to_keep = []
        for edge in self._graph.edges(data=True, keys=True):
            if predicate(edge, item):
                edges_to_keep.append((edge[0], edge[1], edge[2]))
        return self._graph.edge_subgraph(edges_to_keep)

    def add_coordinates(self, coordinates_from, coordinates_to, passangers, timestamp):
        indices_from = self.discretize_coordinates(coordinates_from)
        indices_to = self.discretize_coordinates(coordinates_to)
        if not self._graph.has_node(indices_from):
            self._graph.add_node(indices_from, coordinates=self.node_to_coordinates(indices_from), all_coordinates=[])
        if not self._graph.has_node(indices_to):
            self._graph.add_node(indices_to, coordinates=self.node_to_coordinates(indices_to), all_coordinates=[])

        # self._graph.nodes[indices_from]['all_coordinates'].append(coordinates_from)
        # self._graph.nodes[indices_to]['all_coordinates'].append(coordinates_to)

        self._graph.add_edge(indices_from, indices_to, weight=passangers, timestamp=timestamp)

    def filter_to_weakly_connected(self):
        print 'Creating weakly connected component'
        self._graph = self._graph.subgraph(max(networkx.weakly_connected_components(self._graph), key=len))
        print 'Done'

    def coordinates_to_map(self, coordinates):
        return {'longitude': coordinates[0], 'latitude': coordinates[1]}

    def save_graph(self, path):
        networkx.write_gpickle(self._graph, path)

    def load_graph(self, path):
        self._graph = networkx.read_gpickle(path)


class GraphAnalyser:
    @staticmethod
    def connected_components(graph):
        return (networkx.number_strongly_connected_components(graph),
                networkx.number_weakly_connected_components(graph))

    @staticmethod
    def degree_distribution(graph):
        dic_1 = collections.Counter(sorted([d for n, d in graph.in_degree()]))
        dic_2 = collections.Counter(sorted([d for n, d in graph.out_degree()]))
        return (collections.OrderedDict([(key, dic_1[key]) for key in sorted(dic_1)]),
                collections.OrderedDict([(key, dic_2[key]) for key in sorted(dic_2)]))

    @staticmethod
    def weighted_degree_distribution(graph):
        dic_1 = collections.Counter(sorted([d for n, d in graph.in_degree(weight='weight')]))
        dic_2 = collections.Counter(sorted([d for n, d in graph.out_degree(weight='weight')]))
        return (collections.OrderedDict([(key, dic_1[key]) for key in sorted(dic_1)]),
                collections.OrderedDict([(key, dic_2[key]) for key in sorted(dic_2)]))


def get_degree_distributions():
    jan = Graph()
    jan.load_graph('graph_jan.gpickle')
    sep = Graph()
    sep.load_graph('graph_sep.gpickle')

    print 'Calculating Janury distribution'
    jan_distribution = GraphAnalyser.degree_distribution(jan._graph)
    jan_weighted_distribution = GraphAnalyser.weighted_degree_distribution(jan._graph)

    print 'Calculating September distribution'
    sep_distribution = GraphAnalyser.degree_distribution(sep._graph)
    sep_weighted_distribution = GraphAnalyser.weighted_degree_distribution(sep._graph)

    print 'Calculating in unweighted plot'
    j = graph_drawing.plot_degree_distribution(np.array(jan_distribution[0].keys()), np.array(jan_distribution[0].values()),
                                               'Rozkład stopni wejściowych wierzchołków', 'b')
    s = graph_drawing.plot_degree_distribution(np.array(sep_distribution[0].keys()), np.array(sep_distribution[0].values()),
                                               'Rozkład stopni wejściowych wierzchołków', 'r')
    plt.legend((j, s), (unicode('Styczeń'), unicode('Wrzesień')))
    plt.savefig(os.path.join('results', 'in_degrees.png'))
    plt.clf()

    print 'Calculating out unweighted plot'
    j = graph_drawing.plot_degree_distribution(np.array(jan_distribution[1].keys()), np.array(jan_distribution[1].values()),
                                               'Rozkład stopni wyjściowych wierzchołków', 'b')
    s = graph_drawing.plot_degree_distribution(np.array(sep_distribution[1].keys()), np.array(sep_distribution[1].values()),
                                               'Rozkład stopni wyjściowych wierzchołków', 'r')
    plt.legend((j, s), (unicode('Styczeń'), unicode('Wrzesień')))
    plt.savefig(os.path.join('results', 'out_degrees.png'))
    plt.clf()

    print 'Calculating in weighted plot'
    j = graph_drawing.plot_degree_distribution(np.array(jan_weighted_distribution[0].keys()),
                                               np.array(jan_weighted_distribution[0].values()),
                                               'Rozkład ważonych stopni wejściowych wierzchołków', 'b')
    s = graph_drawing.plot_degree_distribution(np.array(sep_weighted_distribution[0].keys()),
                                               np.array(sep_weighted_distribution[0].values()),
                                               'Rozkład ważonych stopni wejściowych wierzchołków', 'r')
    plt.legend((j, s), (unicode('Styczeń'), unicode('Wrzesień')))
    plt.savefig(os.path.join('results', 'in_weighted_degrees.png'))
    plt.clf()

    print 'Calculating out weighted plot'
    j = graph_drawing.plot_degree_distribution(np.array(jan_weighted_distribution[1].keys()),
                                               np.array(jan_weighted_distribution[1].values()),
                                               'Rozkład ważonych stopni wyjściowych wierzchołków', 'b')
    s = graph_drawing.plot_degree_distribution(np.array(sep_weighted_distribution[1].keys()),
                                               np.array(sep_weighted_distribution[1].values()),
                                               'Rozkład ważonych stopni wyjściowych wierzchołków', 'r')
    plt.legend((j, s), (unicode('Styczeń'), unicode('Wrzesień')))
    plt.savefig(os.path.join('results', 'out_weighted_degrees.png'))
    plt.clf()


def get_maps():
    g = Graph()
    g.load_graph('graph.gpickle')
    print 'Graph order: ', g._graph.order()
    print 'Number of edges: ', g._graph.number_of_edges()
    print 'Number of strongly and weakly connected components: ', GraphAnalyser.connected_components(g._graph)

    g.filter_to_weakly_connected()
    print 'Weak graph order: ', g._graph.order()
    print 'Weak number of edges: ', g._graph.number_of_edges()

    resulting_map = folium.Map(location=g._reference_point, zoom_start=14)
    print 'Creating in heatmap'
    graph_drawing.draw_in_heatmap(g, resulting_map, [(datetime.time(hour=i),
                                                      lambda e, item: item.hour <= e[3]['timestamp'].hour < item.hour + 1)
                                                     for i in range(0, 24)])
    print 'Saving in heatmap'
    resulting_map.save(os.path.join('results', 'heatmap_in_weak.html'))

    resulting_map = folium.Map(location=g._reference_point, zoom_start=14)
    print 'Creating out heatmap'
    graph_drawing.draw_out_heatmap(g, resulting_map, [(datetime.time(hour=i),
                                                       lambda e, item: item.hour <= e[3]['timestamp'].hour < item.hour + 1)
                                                      for i in range(0, 24)])
    print 'Saving out heatmap'
    resulting_map.save(os.path.join('results', 'heatmap_out_weak.html'))

    resulting_map = folium.Map(location=g._reference_point, zoom_start=14)
    print 'Creating edges map'
    graph_drawing.draw_edges(g, resulting_map)
    print 'Saving edges map'
    resulting_map.save(os.path.join('results', 'edges_weak.html'))


def get_clustered_map():
    global path_to_data
    df = pandas.read_csv(path_to_data)
    df = df[df.Cluster != -1]
    resulting_map = folium.Map(location=[df.PickupLatitude.mean(),
                                         df.PickupLongitude.mean()],
                               zoom_start=14)

    color_map = graph_drawing.get_colormap(df.Cluster.min(),
                                           df.Cluster.max(),
                                           graph_drawing.color_map_cluster).add_to(resulting_map)

    df = df.sample(frac=0.01)
    for row in df.itertuples():
        pickup_location = [row.PickupLatitude,
                           row.PickupLongitude]

        dropoff_location = [row.DropoffLatitude,
                            row.DropoffLongitude]

        folium.CircleMarker(pickup_location,
                            color=color_map.rgb_hex_str(row.Cluster),
                            fill=True).add_to(resulting_map)
        folium.CircleMarker(dropoff_location,
                            color=color_map.rgb_hex_str(row.Cluster),
                            fill=True).add_to(resulting_map)

    resulting_map.save(os.path.join('results', 'cluster.html'))


if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')

    get_degree_distributions()
    get_maps()
    get_clustered_map()