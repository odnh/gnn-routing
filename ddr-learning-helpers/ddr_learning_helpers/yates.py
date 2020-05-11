import os
import pathlib
import re
import subprocess

import networkx as nx
import numpy as np
from scipy.special import softmax

Routing = np.ndarray


def nx_to_yates_dot(graph: nx.DiGraph, path: str):
    """
    Converts nx DiGraph following internal ddr standards to one that can be
    understood by yates and writes it to the file at path.
    Args:
        graph: an nx DiGraph
        path: out file path
    Returns:
    """
    g = nx.DiGraph()
    for node in graph.nodes:
        g.add_node("h{}".format(node), type="host", ip="0.0.0.0",
                   mac="00:00:00:00:00:00")
        g.add_node("s{}".format(node), type="switch", id="{}".format(node))
        g.add_edge("h{}".format(node), "s{}".format(node), capacity="1Gbps",
                   cost=1, dst_port=1, src_port=1)
        g.add_edge("s{}".format(node), "h{}".format(node), capacity="1Gbps",
                   cost=1, dst_port=1, src_port=1)
    for (src, dst) in graph.edges:
        g.add_edge("s{}".format(src), "s{}".format(dst), capacity="1Gbps",
                   cost=1, dst_port=1, src_port=1)
    nx.drawing.nx_pydot.write_dot(g, path)


def get_ddr_routing(raw_routing: str, graph: nx.DiGraph) -> np.ndarray:
    """
    Reads the raw output from the routing given by Yates and puts it into the
    ndarray format used by the rest of this codebase
    Args:
        raw_routing: yates routing as a string
    Returns:
        ndarray routing (per flow edge splitting ratios)
    """
    # Set helper data structures
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    flows = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if
             i != j]
    flow_map = {flow: i for i, flow in enumerate(flows)}
    edge_map = {edge: i for i, edge in enumerate(sorted(graph.edges))}

    # initialise empty routing
    routing = np.zeros((len(flows), num_edges), dtype=np.float32)

    # initialise loop vars
    current_flow = (0, 0)

    # create the regexes to parse the lines
    match_flow_line = re.compile('->')  # line that defines a flow
    match_edges_line = re.compile('@')  # line that lists edges in a path
    match_src_dst = re.compile(r'\d+')  # get the values from a flow definition
    match_edges = re.compile(r'\(s\d+,s\d+\)')  # get (valid) edges from path
    match_edge_ends = re.compile(r'\d+')  # get the node ids from an edge
    match_value = re.compile(r'\d.\d+')  # get split ratio for a path

    lines = raw_routing.split("\n")
    for line in lines:
        # match line at start of path section defining which flow they are for
        if match_flow_line.search(line) is not None:
            src_dst = match_src_dst.findall(line)
            current_flow = (int(src_dst[0]), int(src_dst[1]))
        # match that this is a line defining one of the possible flows
        elif match_edges_line.search(line) is not None:
            edges = match_edges.findall(line)
            edge_ends = [match_edge_ends.findall(edge) for edge in edges]
            value = match_value.findall(line)
            for edge_str in edge_ends:
                edge = (int(edge_str[0]), int(edge_str[1]))
                if edge[0] != edge[1]:
                    routing[flow_map[current_flow]][edge_map[edge]] += float(
                        value[0])

    # due to the algorithm setting proportions of flow on paths rather than
    # splitting ratios for each node we have to rescale the ratios on each edge
    # at each node to add to one
    # NB: the Raeke algorithms sometimes creates loops in small graphs so this
    # may lead to underestimates where this is the case
    normalised_routing = np.zeros((len(flows), num_edges), dtype=np.float32)
    for flow_idx in range(len(flows)):
        for node_idx in range(num_nodes):
            out_edges = graph.out_edges(node_idx)
            out_edge_ids = [edge_map[edge] for edge in out_edges]
            out_edge_weights = [routing[flow_idx][edge_idx] for edge_idx in
                                out_edge_ids]
            if np.sum(out_edge_weights) != 0.0:
                normalised_out_edge_weights = np.divide(out_edge_weights, np.sum(out_edge_weights))
            else:
                normalised_out_edge_weights = out_edge_weights
            for i, edge_idx in enumerate(out_edge_ids):
                normalised_routing[flow_idx][edge_idx] = \
                    normalised_out_edge_weights[i]

    return normalised_routing


def get_oblivious_routing(graph: nx.DiGraph) -> Routing:
    """
    Takes a graph, runs it through the Yates Raeke implementation and returns
    an oblivious routing as an ndarray
    Args:
        graph: nx DiGraph to calculate the routing for
    Returns:
        An oblivious routing as an ndarray
    """
    script_path = pathlib.Path(__file__).parent.absolute()
    oblivious_tmp_path = "{}/../../data/oblivious/tmp_dot".format(script_path)
    raeke_path = "{}/../../raeke/_build/install/default/bin/raeke_route".format(
        script_path)
    # create tmp dot file for yates
    nx_to_yates_dot(graph, oblivious_tmp_path)
    # run yates to get the routing
    raw_routing = subprocess.run([raeke_path, oblivious_tmp_path],
                                 stdout=subprocess.PIPE).stdout.decode('utf-8')
    ddr_routing = get_ddr_routing(raw_routing, graph)
    # os.remove(oblivious_tmp_path)

    return ddr_routing
