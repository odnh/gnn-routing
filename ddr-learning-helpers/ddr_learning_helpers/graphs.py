import pathlib

import networkx as nx
import numpy as np
from ddr_learning_helpers import totem


def random(num_nodes: int,
           num_edges: int,
           weight: int = 10000) -> nx.OrderedDiGraph:
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
    Fairly simple graph for testing
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


def basic2() -> nx.OrderedDiGraph:
    """
    Very simple graph for testing
    Returns:
        A simple graph
    """
    graph = nx.OrderedDiGraph()
    graph.add_nodes_from([0, 1, 2, 3])
    graph.add_edge(0, 1, weight=5000)
    graph.add_edge(1, 0, weight=5000)
    graph.add_edge(0, 2, weight=5000)
    graph.add_edge(2, 0, weight=5000)
    graph.add_edge(0, 3, weight=5000)
    graph.add_edge(3, 0, weight=5000)
    graph.add_edge(1, 2, weight=5000)
    graph.add_edge(2, 1, weight=5000)
    graph.add_edge(2, 3, weight=5000)
    graph.add_edge(3, 2, weight=5000)
    return graph


def full() -> nx.OrderedDiGraph:
    """Small-ish fully-connected graph"""
    graph = nx.complete_graph(5, nx.OrderedDiGraph)
    for src, dst in graph.edges():
        graph[src][dst] = 10000
    return graph


def from_graphspec(graphspec: str) -> nx.DiGraph:
    """
    graphspec: topologyzooname:n/e:+/-:seed
    i.e. can select by name and randomly drop/add an edge/node
    """
    weight = 10000
    parsed = graphspec.split(":")
    name = parsed[0]
    graph = topologyzoo(name, weight)

    # overrides for non-topologyzoo graphs
    if name == 'basic':
        graph = basic()
    elif name == 'basic2':
        graph = basic2()
    elif name == 'totem':
        t = totem.Totem(weight=10000)
        graph = t.graph
    elif name == 'full':
        graph = full()

    # graphspec allows edges and nodes to be dropped or added
    if len(parsed) > 1:
        seed = int(parsed[3])
        random_state = np.random.RandomState(seed)
        if parsed[1] == 'e':
            if parsed[2] == '+':
                node_a = random_state.randint(0, graph.number_of_nodes())
                node_b = random_state.randint(0, graph.number_of_nodes())
                if node_a == node_b:
                    node_b = (node_a + 1) % graph.number_of_nodes()
                if not graph.has_edge(node_a, node_b):
                    graph.add_edge(node_a, node_b, weight=weight)
                    graph.add_edge(node_b, node_a, weight=weight)
            elif parsed[2] == '-':
                edge_idx = random_state.randint(0, graph.number_of_edges())
                edge = list(graph.edges())[edge_idx]
                graph.remove_edge(edge[0], edge[1])
                graph.remove_edge(edge[1], edge[0])
            else:
                raise Exception("Invalid graphspec")
        elif parsed[1] == 'n':
            if parsed[2] == '+':
                node_a = graph.number_of_nodes()
                node_b = random_state.randint(0, graph.number_of_nodes())
                graph.add_node(graph.number_of_nodes())
                graph.add_edge(node_a, node_b, weight=weight)
                graph.add_edge(node_b, node_a, weight=weight)
            elif parsed[2] == '-':
                node = random_state.randint(0, graph.number_of_nodes())
                graph.remove_node(node)
            else:
                raise Exception("Invalid graphspec")
        else:
            raise Exception("Invalid graphspec")

        graph = nx.convert_node_labels_to_integers(graph)

    return graph
