#!/usr/bin/python2
# encoding=utf-8

import branca
import collections
import folium
import matplotlib.pyplot as plt
import numpy as np
import sys

color_map_heatmap = collections.OrderedDict(sorted({0.01: "blue",
                                                    0.05: "cyan",
                                                    0.1: "lime",
                                                    0.5: "yellow",
                                                    1.0: "red"}.items()))

color_map_cluster = collections.OrderedDict(sorted({0.0: 'blue',
                                                    1.0: 'red'}.items()))


def get_colormap(min_degree, max_degree, color_map):
    return branca.colormap.LinearColormap(colors=color_map.values(),
                                          index=[max_degree * value for value in color_map.keys()],
                                          vmin=min_degree,
                                          vmax=max_degree)


def normalize(data):
    max_value = max([float(max(z, key=lambda entry: entry[2])[2]) for z in data])
    for first in data:
        for second in first:
            second[2] /= max_value
    return data


def draw_in_heatmap(graph, map, predicates):
    ordered_data = collections.OrderedDict()
    # for time in [datetime.time(hour=i) for i in range(0, 24)]:
    for item, predicate in predicates:
        data = []
        for node in graph.filter_edges(predicate, item).nodes(data=True):
            data.append([node[1]['coordinates'][0],
                         node[1]['coordinates'][1],
                         graph._graph.in_degree[node[0]]])
        ordered_data[item] = data

    normalize(ordered_data.values())

    return folium.plugins.HeatMapWithTime(ordered_data.values(),
                                          index=[str(t) for t in ordered_data.keys()],
                                          min_opacity=0.2,
                                          radius=15).add_to(map)


def draw_out_heatmap(graph, map, predicates):
    ordered_data = collections.OrderedDict()
    # for time in [datetime.time(hour=i) for i in range(0, 24)]:
    for item, predicate in predicates:
        data = []
        for node in graph.filter_edges(predicate, item).nodes(data=True):
            data.append([node[1]['coordinates'][0],
                         node[1]['coordinates'][1],
                         graph._graph.out_degree[node[0]]])
        ordered_data[item] = data

    normalize(ordered_data.values())

    return folium.plugins.HeatMapWithTime(ordered_data.values(),
                                          index=[str(t) for t in ordered_data.keys()],
                                          min_opacity=0.2,
                                          radius=15).add_to(map)


def draw_edges(graph, map):
    reload(sys)
    sys.setdefaultencoding('utf8')

    edges = {}
    for u, v, key, data in graph._graph.edges(data=True, keys=True):
        if (u, v) in edges:
            edges[u, v] += data['weight']
        else:
            edges[u, v] = data['weight']

    min_weight = 100
    colormap = get_colormap(min_weight, max(edges.values()), color_map_heatmap)
    colormap.caption = 'Liczba ludzi jeżdżących między obszarami'
    colormap.add_to(map)
    for edge, weight in edges.items():
        if weight >= min_weight:
            # print weight
            # print colormap.rgb_hex_str(weight)
            attr = {'fill': colormap.rgb_hex_str(weight), 'font-weight': 'bold', 'font-size': '12'}
            coordinates = [graph._graph.nodes[edge[0]]['coordinates'], graph._graph.nodes[edge[1]]['coordinates']]
            line = folium.features.PolyLine(coordinates,
                                            weight=10, color=colormap.rgb_hex_str(weight), opacity=0.2).add_to(map)
            folium.plugins.PolyLineTextPath(line, "  ►  ", repeat=True, offset=8, attributes=attr).add_to(map)


def plot_degree_distribution(degrees, counts, title, color):
    first_element = 0
    if degrees[0] == 0:
        first_element = 1

    counts = np.array([counts[i:].sum() for i in range(len(counts))])

    # Plot
    s = plt.scatter(degrees[first_element:], counts[first_element:], s=1, color=color)
    plt.title(unicode(title))
    plt.ylabel(unicode('Liczba wierzchołków o stopniu >= V'))
    plt.xlabel(unicode('Stopień wierzchołków V'))

    plt.xscale('log')
    plt.yscale('log')

    return s