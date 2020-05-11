import networkx as nx
import numpy as np
from ddr_learning_helpers import yates


def normalise_array(array: np.ndarray) -> np.ndarray:
    """
    Takes an array of positive values and normalises them so they sum to one
    Args:
        array: np array
    Returns:
        normalised np array
    """
    return np.divide(array, np.sum(array)) if np.sum(array) != 0 else array


def shortest_path_routing(graph: nx.DiGraph) -> np.ndarray:
    """
    Creates a ddr routing for a graph using shortest path routing
    Args:
        graph: graph to route over
    Returns:
        nd array containing the splitting ratios
    """
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    flows = [(i, j) for i in range(num_nodes) for j in range(num_nodes)
             if i != j]
    edge_order = {edge: i for i, edge in enumerate(sorted(graph.edges))}

    routing = np.zeros((len(flows), num_edges))
    for flow_idx, (src, dst) in enumerate(flows):
        route = nx.shortest_path(graph, src, dst)
        for edge_source_idx in range(len(route) - 1):
            edge_idx = edge_order[
                (route[edge_source_idx], route[edge_source_idx + 1])]
            routing[flow_idx][edge_idx] = 1.0

    return routing


def random_routing(graph: nx.DiGraph) -> np.ndarray:
    """
    Creates a ddr routing with random splitting ratios
    Args:
        graph: graph to route over
    Returns:
        nd array with splitting ratios that are valid (but random)
    """
    out_edge_count = np.cumsum(
        [i[1] for i in sorted(graph.out_degree(), key=lambda x: x[0])])
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    num_demands = graph.number_of_nodes() * \
                  (graph.number_of_nodes() - 1)

    routing = np.random.random((num_demands, num_edges))
    normalised_routing = np.zeros((num_demands, num_edges))
    for flow_idx in range(num_demands):
        for intermediate_node in range(num_nodes):
            start_idx = out_edge_count[
                intermediate_node - 1] if intermediate_node - 1 >= 0 else 0
            end_idx = out_edge_count[intermediate_node]
            normalised_routing[flow_idx][start_idx:end_idx] = normalise_array(
                routing[flow_idx][start_idx:end_idx])
    return normalised_routing


def raeke_routing(graph: nx.DiGraph) -> np.ndarray:
    """
    Creates a ddr routing using Raeke's algorithm (using code from the Yates
    project)
    Args:
        graph: graph to route over
    Returns:
        nd array with splitting ratios
    """
    return yates.get_oblivious_routing(graph)