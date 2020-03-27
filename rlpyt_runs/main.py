import argparse

import networkx as nx

from rlpyt_runs import graphs
from rlpyt_runs.experiments import mlp


def get_graph(graph_spec: str) -> nx.DiGraph:
    graph_spec = graph_spec.split(':')
    if graph_spec[0] == 'basic':
        return graphs.generator.basic()
    elif graph_spec[0] == 'zoo':
        return graphs.generator.topologyzoo(graph_spec[1], int(graph_spec[2]))
    elif graph_spec[0] == 'random':
        return graphs.generator.random(
            int(graph_spec[1]), int(graph_spec[2]), int(graph_spec[3]))
    else:
        print("Not a valid graph spec")


def dispatch(experiment_name: str, graph_spec: str):
    graph = get_graph(graph_spec)
    if experiment_name == "mlp_dest":
        mlp.run_experiment("dest", graph)
    elif experiment_name == "mlp_soft":
        mlp.run_experiment("softmin", graph)
    elif experiment_name == "gnn_basic":
        pass
    else:
        print("No such experiment", experiment_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", help="name of experiment to run",
                        type=str)
    parser.add_argument("graph", help="graph to run experiment on",
                        type=str)
    args = parser.parse_args()
    dispatch(args.experiment, args.graph)
