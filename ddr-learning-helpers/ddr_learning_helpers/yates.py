import os
import pathlib
import re
import subprocess

import networkx as nx
import numpy as np

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

    # create the regexes
    match_flow_line = re.compile('->')
    match_edges_line = re.compile('@')
    match_src_dst = re.compile(r'\d+')
    match_edges = re.compile(r'\d+(?=,|\))')
    match_value = re.compile(r'\d.\d+')

    lines = raw_routing.split("\n")

    for line in lines:
        if match_flow_line.search(line) is not None:
            src_dst = match_src_dst.findall(line)
            current_flow = (int(src_dst[0]), int(src_dst[1]))
        elif match_edges_line.search(line) is not None:
            edges = match_edges.findall(line)
            value = match_value.findall(line)
            for i in range(0, len(edges), 2):
                # set routing
                edge = (int(edges[i]), int(edges[i + 1]))
                if edge[0] == edge[1]: continue
                routing[flow_map[current_flow]][edge_map[edge]] = float(
                    value[0])

    return routing


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
    os.remove(oblivious_tmp_path)

    return ddr_routing
