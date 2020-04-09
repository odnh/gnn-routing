import networkx as nx
import numpy as np
import pathlib

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
    nx.drawing.nx_pydot.write_dot(g)


def get_oblivious_routing(graph: nx.DiGraph) -> Routing:
    script_path = pathlib.Path(__file__).parent.absolute()
    graph = nx.readwrite.read_graphml(
        "{}/../../data/topologyzoo/{}.graphml".format(script_path, graph_name))
    dot_graph = nx_to_yates_dot(graph)

    pass
