import pathlib

import networkx as nx
import numpy as np


def random(num_nodes: int,
           num_edges: int,
           weight: int = 1000) -> nx.OrderedDiGraph:
    """
    Generates a random undirected networkx DiGraph (as in always same edge both
    ways)
    Args:
        num_nodes: number of nodes
        num_edges: number of edges
        weight: weight to be assigned to all edges

    Returns:
        A DiGraph generated as described
    """
    graph = nx.OrderedGraph()
    graph.add_nodes_from(range(num_nodes))
    for _ in range(num_edges):
        src = np.random.randint(0, num_nodes)
        dst = np.random.randint(0, num_nodes)
        graph.add_edge(src, dst, weight=weight)
    return graph.to_directed()


def topologyzoo(graph_name: str, weight: int) -> nx.OrderedDiGraph:
    """
    Reads in graphs from the topologyzoo dataset and puts them in a normalised
    format. i.e. all node names become contiguous numbers and weight is added to
    each link
    Args:
        graph_name: Name of the graph to load
        weight: weight to use on edges

    Returns:
        A DiGraph with weights
    """
    script_path = pathlib.Path(__file__).parent.absolute()
    graph = nx.readwrite.read_graphml(
        "{}/../../data/topologyzoo/{}.graphml".format(script_path, graph_name))
    normalised_graph = nx.OrderedGraph()

    node_mapping = {node: i for i, node in enumerate(graph.nodes())}

    # Add nodes in order
    normalised_graph.add_nodes_from(range(graph.number_of_nodes()))
    # Add edges
    for src, dst in graph.edges():
        normalised_graph.add_edge(
            node_mapping[src], node_mapping[dst], weight=weight)

    return normalised_graph.to_directed()


def basic() -> nx.OrderedDiGraph:
    """
    Very simple graph for testing
    Returns:
        A simple graph
    """
    graph = nx.OrderedDiGraph()
    graph.add_nodes_from([0, 1, 2, 3, 4, 5])
    graph.add_edge(0, 1, weight=5000)
    graph.add_edge(1, 0, weight=5000)
    graph.add_edge(0, 5, weight=5000)
    graph.add_edge(5, 0, weight=5000)
    graph.add_edge(1, 2, weight=5000)
    graph.add_edge(2, 1, weight=5000)
    graph.add_edge(1, 5, weight=5000)
    graph.add_edge(5, 1, weight=5000)
    graph.add_edge(2, 4, weight=5000)
    graph.add_edge(4, 2, weight=5000)
    graph.add_edge(2, 5, weight=5000)
    graph.add_edge(5, 2, weight=5000)
    graph.add_edge(3, 4, weight=5000)
    graph.add_edge(4, 3, weight=5000)
    graph.add_edge(4, 5, weight=5000)
    graph.add_edge(5, 4, weight=5000)
    return graph
