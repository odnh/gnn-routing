from typing import Tuple

import networkx as nx
import numpy as np
from ortools.linear_solver import pywraplp

Demand = np.ndarray
Routing = np.ndarray


def opt(graph: nx.DiGraph, demands: Demand) -> float:
    """
    Returns the optimal minimum of max-link-utilisation for the graph, given
    the demands
    Args:
      graph: network to optimise over (must be ph with all edges bidirectional)
      demands: demand array should be 1 x (|V|*(|V|-1)) in number of nodes
    Returns:
      min max-link-utilisation
    """
    ## Build helper data
    edges = list(sorted(graph.edges()))
    edge_index_dict = {edge: i for i, edge in enumerate(edges)}

    # create commodities from demands (make sure to ignore demand to self)
    commodities = []
    flow_count = 0
    for i in range(graph.number_of_nodes()):
        for j in range(graph.number_of_nodes()):
            if i != j:
                commodities.append((i, j, demands[flow_count]))
                flow_count += 1

    # Create the linear solver with the GLOP backend.
    solver = pywraplp.Solver('multicommodity_flow_lp',
                             pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

    ## VARIABLES
    # Flow variables, the splitting ratios for each edge
    # Stored as a list of lists (flow_variables[ith_flow][jth_edge])
    flow_variables = []
    for i in range(len(commodities)):
        flow_variable_edges = []
        for j in range(graph.number_of_edges()):
            flow_variable_edges.append(
                solver.NumVar(0, 1, '({},{})'.format(i, j)))
        flow_variables.append(flow_variable_edges)

    ## CONSTRAINTS
    # Capacity constraint
    capacity_constraints = []
    for i, edge in enumerate(edges):
        # Constraint between 0 and edge capacity
        constraint_i = solver.Constraint(
            0, graph.get_edge_data(*edge)['weight'], '(1,{},{})'.format(*edge))
        for j, commodity in enumerate(commodities):
            # Coefficient for jth flow over ith edge is scaled by flow width
            constraint_i.SetCoefficient(flow_variables[j][i],
                                        # cast because or-tools :'(
                                        float(commodity[2]))
        capacity_constraints.append(constraint_i)

    # Conservation on transit nodes
    conservation_transit_constraints = []
    for i, commodity in enumerate(commodities):
        constraints_flow_i = []
        for j in range(graph.number_of_nodes()):
            if j != commodity[0] and j != commodity[1]:
                # Constraint must sum to zero
                constraint_j = solver.Constraint(0, 0, '(2,{},{})'.format(i, j))
                for k in list(sorted(graph.adj[j].keys())):
                    # Ingress edges
                    constraint_j.SetCoefficient(
                        flow_variables[i][edge_index_dict[(k, j)]], 1)
                    # Egress edges
                    constraint_j.SetCoefficient(
                        flow_variables[i][edge_index_dict[(j, k)]], -1)
                constraints_flow_i.append(constraint_j)
        conservation_transit_constraints.append(constraints_flow_i)

    # Conservation  of flow at source node
    conservation_source_constraints = []
    for i, commodity in enumerate(commodities):
        # Constraint must sum to one (assuming all the demand can be met)
        constraint_i = solver.Constraint(1, 1, '(3,{})'.format(i))
        for edge_dest in list(sorted(graph.adj[commodity[0]].keys())):
            constraint_i.SetCoefficient(
                flow_variables[i][edge_index_dict[(commodity[0], edge_dest)]],
                1)
            constraint_i.SetCoefficient(
                flow_variables[i][edge_index_dict[(edge_dest, commodity[0])]],
                -1)
        conservation_source_constraints.append(constraint_i)

    # Conservation of flow at destination node
    conservation_dest_constraints = []
    for i, commodity in enumerate(commodities):
        # Constraint must sum to one (assuming all the demand can be met)
        constraint_i = solver.Constraint(1, 1, '(4,{})'.format(i))
        for edge_dest in list(sorted(graph.adj[commodity[1]].keys())):
            constraint_i.SetCoefficient(
                flow_variables[i][edge_index_dict[(edge_dest, commodity[1])]],
                1)
            constraint_i.SetCoefficient(
                flow_variables[i][edge_index_dict[(commodity[1], edge_dest)]],
                -1)
        conservation_dest_constraints.append(constraint_i)

    ## OBJECTIVES
    # Implementation of the load-balancing example from wikipedia
    # First we add more constraints so that we are minimising the maximum
    max_utilisation_variable = solver.NumVar(0, solver.Infinity(),
                                             'max_link_utilisation')
    min_of_max_constraints = []
    for i, edge in enumerate(edges):
        # Constraint that '-inf < f_0 + f_1 +... - max < 0'
        # i.e 'f_0 + f_1 + ... < max'
        constraint_i = solver.Constraint(-solver.Infinity(), 0,
                                         '(5,{})'.format(i))
        constraint_i.SetCoefficient(max_utilisation_variable, -1)
        for j, flow_variable in enumerate(flow_variables):
            constraint_i.SetCoefficient(flow_variable[i],
                                        commodities[j][2] /
                                        graph.get_edge_data(*edge)['weight'])
        min_of_max_constraints.append(constraint_i)

    # Objective now is to minimise the maximum link utilisation
    objective = solver.Objective()
    objective.SetCoefficient(max_utilisation_variable, 1)
    objective.SetMinimization()
    solver.Solve()

    return objective.Value()

"""
Algorithm Planning
end result: array of bandwidth used on each edge
high-level: calculate such an array for every flow and sum them
per-flow:
  1. initialise entire demand at start node
  2. push bandwidth onto edges and neighbour nodes (delete from self)
  3. perform 2 for all nodes what were changed in prev step
  3.1 NB: if dst node, then just absorb the flow rather than push on
  4. end when no node changes value (may need some min float limit)
"""


def calc_per_flow_link_utilisation(graph: nx.DiGraph, flow: Tuple[int, int],
                                   demand: float,
                                   routing: np.ndarray) -> np.ndarray:
    """
    Calculates the link utilisation over a graph for a particular flow and its
    demand. (NB utilisation in bandwidth, not relative to capacity)
    Args:
        graph: network graph to calculate over
        flow: the source and destination node ids of the flow
        demand: the bandwidth to push from source to sink
        routing: edge splitting ratios to be used when routing
    Returns:
        Link utilisation as an ndarray (one value per edge)
    """
    # TODO: make this code run much faster
    num_edges = graph.number_of_edges()
    num_nodes = graph.number_of_nodes()
    edge_mapping = {edge: i for i, edge in enumerate(sorted(graph.edges))}

    link_utilisation = np.zeros(num_edges)
    node_flow = np.zeros(num_nodes)  # the flow stored at a node
    node_flow[flow[0]] = demand

    to_explore = [flow[0]]
    while to_explore:
        current_node = to_explore.pop(0)
        current_flow = node_flow[current_node]

        # nothing to do here then so don't even try
        if np.isclose(current_flow, 0.0):  # TODO: fix this to be better
            continue

        # this is the flow destination node so we absorb all flow
        if current_node == flow[1]:
            node_flow[current_node] = 0.0
            continue

        # push the flow at this node over all edges
        for edge in graph.out_edges(current_node):
            edge_index = edge_mapping[edge]
            ratio = routing[edge_index]
            node_flow[edge[1]] += ratio * current_flow
            # all important step, update our output
            link_utilisation[edge_index] += ratio * current_flow
            # have updated the dst so need to add it to the list of things to do
            to_explore.append(edge[1])
        # we've moved all the flow from this node now, so reset back to zero
        node_flow[current_node] = 0.0

    return link_utilisation


def calc_overall_link_utilisation(graph: nx.DiGraph, demands: Demand,
                                  routing: Routing) -> np.ndarray:
    """
    Calculates the overall utilisation of each link in a network given a routing
    choice and a set of demands. (NB utilisation in bandwidth, not relative to
    capacity)
    Args:
        graph: network graph to calculate over
        demands: demands matrix, format is: 1 x (|V|*(|V|-1)) (ordering is
                 (0,1),(0,2),...,(1,0),(1,2),... etc)
        routing: per-flow edge splitting rations, format is (|V|*(|V|-1)) x |E|
                 (same ordering as flows, edges numerically ordered)
    Returns:
        Link utilisation as an ndarray (one value per edge)
    """
    num_edges = graph.number_of_edges()
    num_nodes = graph.number_of_nodes()
    flows = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if
             i != j]

    link_utilisation = np.zeros(num_edges)

    for i, flow in enumerate(flows):
        flow_link_utilisation = calc_per_flow_link_utilisation(graph, flow,
                                                               demands[i],
                                                               routing[i])
        link_utilisation += flow_link_utilisation

    return link_utilisation


def calc(graph: nx.DiGraph, demands: Demand, routing: Routing) -> float:
    """
    Returns the max-link-utilisation for the graph, given
    the demands and routing
    Args:
      graph: network to optimise over
      demands: demand matrix, should be 1 x (|V|*(|V|-1)) in number of nodes
      routing: per-flow edge-splitting ratios (|V|*(|V|-1)) x |E|
    Returns:
      max-link-utilisation

    NB: does not actually check given routing is valid but assumes it is.
    """
    edge_capacities = [e[2]['weight'] for e in sorted(graph.edges(data=True))]
    link_utilisation = calc_overall_link_utilisation(graph, demands, routing)
    # Because utilisation compared to link width is what we care about here
    ratio_capacities = np.divide(link_utilisation, edge_capacities)

    return np.max(ratio_capacities)

