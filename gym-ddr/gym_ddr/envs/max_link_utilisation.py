from typing import Tuple

import networkx as nx
import numpy as np
from ortools.linear_solver import pywraplp

Demand = np.ndarray
Routing = np.ndarray


class MaxLinkUtilisation:
    """
    Class for calculating all things relating to max link utilisation over a
    graph. Can find optimal routing for a set of demands or calculate the
    utilisation gfiv
    """

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self.num_nodes = graph.number_of_nodes()
        self.num_edges = graph.number_of_edges()

        # Build helper data
        self.edges = list(sorted(graph.edges(data=True)))
        self.edge_index_dict = {(edge[0], edge[1]): i for i, edge in
                                enumerate(self.edges)}
        self.commodities = [(i, j) for i in range(self.num_nodes) for j in
                            range(self.num_nodes) if i != j]

    def opt(self, demands: Demand) -> float:
        """
        Returns the optimal minimum of max-link-utilisation for the graph, given
        the demands
        Args:
          demands: demand array should be 1 x (|V|*(|V|-1)) in number of nodes
        Returns:
          min max-link-utilisation
        """
        # Create the linear solver with the GLOP backend.
        solver = pywraplp.Solver('multicommodity_flow_lp',
                                 pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

        ## VARIABLES
        # Flow variables, the splitting ratios for each edge
        # Stored as a list of lists (flow_variables[ith_flow][jth_edge])
        flow_variables = []
        for i in range(len(self.commodities)):
            flow_variable_edges = []
            for j in range(self.num_edges):
                flow_variable_edges.append(
                    solver.NumVar(0, 1, '({},{})'.format(i, j)))
            flow_variables.append(flow_variable_edges)

        ## CONSTRAINTS
        # Capacity constraint
        capacity_constraints = []
        for i, edge in enumerate(self.edges):
            # Constraint between 0 and edge capacity
            constraint_i = solver.Constraint(
                0, self.graph.get_edge_data(*edge)['weight'],
                '(1,{},{})'.format(*edge))
            for j, commodity in enumerate(self.commodities):
                # Coefficient for jth flow over ith edge is scaled by flow width
                constraint_i.SetCoefficient(flow_variables[j][i],
                                            # cast because or-tools :'(
                                            float(demands[j]))
            capacity_constraints.append(constraint_i)

        # Conservation on transit nodes
        conservation_transit_constraints = []
        for i, commodity in enumerate(self.commodities):
            constraints_flow_i = []
            for j in range(self.num_nodes):
                if j != commodity[0] and j != commodity[1]:
                    # Constraint must sum to zero
                    constraint_j = solver.Constraint(0, 0,
                                                     '(2,{},{})'.format(i, j))
                    for k in list(sorted(self.graph.adj[j].keys())):
                        # Ingress edges
                        constraint_j.SetCoefficient(
                            flow_variables[i][self.edge_index_dict[(k, j)]], 1)
                        # Egress edges
                        constraint_j.SetCoefficient(
                            flow_variables[i][self.edge_index_dict[(j, k)]], -1)
                    constraints_flow_i.append(constraint_j)
            conservation_transit_constraints.append(constraints_flow_i)

        # Conservation of flow at source node
        conservation_source_constraints = []
        for i, commodity in enumerate(self.commodities):
            # Constraint must sum to one (assuming all the demand can be met)
            constraint_i = solver.Constraint(1, 1, '(3,{})'.format(i))
            for edge_dest in list(sorted(self.graph.adj[commodity[0]].keys())):
                constraint_i.SetCoefficient(
                    flow_variables[i][
                        self.edge_index_dict[(commodity[0], edge_dest)]],
                    1)
                constraint_i.SetCoefficient(
                    flow_variables[i][
                        self.edge_index_dict[(edge_dest, commodity[0])]],
                    -1)
            conservation_source_constraints.append(constraint_i)

        # Conservation of flow at destination node
        conservation_dest_constraints = []
        for i, commodity in enumerate(self.commodities):
            # Constraint must sum to one (assuming all the demand can be met)
            constraint_i = solver.Constraint(1, 1, '(4,{})'.format(i))
            for edge_dest in list(sorted(self.graph.adj[commodity[1]].keys())):
                constraint_i.SetCoefficient(
                    flow_variables[i][
                        self.edge_index_dict[(edge_dest, commodity[1])]],
                    1)
                constraint_i.SetCoefficient(
                    flow_variables[i][
                        self.edge_index_dict[(commodity[1], edge_dest)]],
                    -1)
            conservation_dest_constraints.append(constraint_i)

        ## OBJECTIVES
        # Implementation of the load-balancing example from Wikipedia
        # First we add more constraints so that we are minimising the maximum
        max_utilisation_variable = solver.NumVar(0, solver.Infinity(),
                                                 'max_link_utilisation')
        min_of_max_constraints = []
        for i, edge in enumerate(self.edges):
            # Constraint that '-inf < f_0 + f_1 +... - max < 0'
            # i.e 'f_0 + f_1 + ... < max'
            constraint_i = solver.Constraint(-solver.Infinity(), 0,
                                             '(5,{})'.format(i))
            constraint_i.SetCoefficient(max_utilisation_variable, -1)
            for j, flow_variable in enumerate(flow_variables):
                constraint_i.SetCoefficient(flow_variable[i],
                                            demands[j] /
                                            self.graph.get_edge_data(*edge)[
                                                'weight'])
            min_of_max_constraints.append(constraint_i)

        # Objective now is to minimise the maximum link utilisation
        objective = solver.Objective()
        objective.SetCoefficient(max_utilisation_variable, 1)
        objective.SetMinimization()
        solver.Solve()

        # # extract the actual routing. Useful for debugging, maybe use to bootstrap
        # opt_routing = np.zeros((len(commodities), self.num_edges))
        # for i in range(len(commodities)):
        #     for j in range(self.num_edges):
        #         opt_routing[i][j] = flow_variables[i][j].solution_value()
        # print(opt_routing)

        return objective.Value()

    def calc_per_flow_link_utilisation(self, flow: Tuple[int, int],
                                       demand: float,
                                       routing: np.ndarray) -> np.ndarray:
        """
        Calculates the link utilisation over a graph for a particular flow and its
        demand. (NB utilisation in bandwidth, not relative to capacity)
        Args:
            flow: the source and destination node ids of the flow
            demand: the bandwidth to push from source to sink
            routing: edge splitting ratios to be used when routing
        Returns:
            Link utilisation as an ndarray (one value per edge)
        """
        # TODO: make this code run much faster
        edge_mapping = {edge: i for i, edge in
                        enumerate(sorted(self.graph.edges))}

        link_utilisation = np.zeros(self.num_edges)
        node_flow = np.zeros(self.num_nodes)  # the flow stored at a node
        node_flow[flow[0]] = demand

        to_explore = [flow[0]]
        while to_explore:
            current_node = to_explore.pop(0)
            current_flow = node_flow[current_node]

            # this is the flow destination node so we absorb all flow
            if current_node == flow[1]:
                node_flow[current_node] = 0.0
                continue

            # push the flow at this node over all edges
            for edge in self.graph.out_edges(current_node):
                edge_index = edge_mapping[edge]
                ratio = routing[edge_index]
                flow_to_send = ratio * current_flow
                # only send flow if greater than epsilon (so no 'infinite' loops)
                if flow_to_send > 1.e-8:  # TODO: can we do this better?
                    node_flow[edge[1]] += ratio * current_flow
                    # all important step, update our output
                    link_utilisation[edge_index] += ratio * current_flow
                    # have updated the dst so add it to the list of things to do
                    to_explore.append(edge[1])
            # we've moved all the flow from this node now, so reset back to zero
            node_flow[current_node] = 0.0

        return link_utilisation

    def calc_overall_link_utilisation(self, demands: Demand,
                                      routing: Routing) -> np.ndarray:
        """
        Calculates the overall utilisation of each link in a network given a routing
        choice and a set of demands. (NB utilisation in bandwidth, not relative to
        capacity)
        Args:
            demands: demands matrix, format is: 1 x (|V|*(|V|-1)) (ordering is
                     (0,1),(0,2),...,(1,0),(1,2),... etc)
            routing: per-flow edge splitting rations, format is (|V|*(|V|-1)) x |E|
                     (same ordering as flows, edges numerically ordered)
        Returns:
            Link utilisation as an ndarray (one value per edge)
        """
        flows = [(i, j) for i in range(self.num_nodes)
                 for j in range(self.num_nodes)
                 if i != j]

        link_utilisation = np.zeros(self.num_edges)

        for i, flow in enumerate(flows):
            flow_link_utilisation = self.calc_per_flow_link_utilisation(flow,
                                                                        demands[
                                                                            i],
                                                                        routing[
                                                                            i])
            link_utilisation += flow_link_utilisation

        return link_utilisation

    def slow_calc(self, demands: Demand, routing: Routing) -> float:
        """
        Returns the max-link-utilisation for the graph, given
        the demands and routing
        Args:
          demands: demand matrix, should be 1 x (|V|*(|V|-1)) in number of nodes
          routing: per-flow edge-splitting ratios (|V|*(|V|-1)) x |E|
        Returns:
          max-link-utilisation
        """
        edge_capacities = [e[2]['weight'] for e in
                           sorted(self.graph.edges(data=True))]
        link_utilisation = self.calc_overall_link_utilisation(demands, routing)
        # Because utilisation compared to link width is what we care about here
        ratio_capacities = np.divide(link_utilisation, edge_capacities)

        return np.max(ratio_capacities)

    def calc(self, demands: Demand, routing: Routing) -> float:
        """
        Returns the max-link-utilisation for the graph, given
        the demands and routing
        This method is hopefully much faster (by using LP rather than naive flow
        pushing).
        Args:
          demands: demand matrix, should be 1 x (|V|*(|V|-1)) in number of nodes
          routing: per-flow edge-splitting ratios (|V|*(|V|-1)) x |E|
        Returns:
          max-link-utilisation
        """
        epsilon = 1.e-2  # TODO: work out how to make this more stable (objective maybe)

        # Create the linear solver with the GLOP backend.
        solver = pywraplp.Solver('flow_utilisation_lp',
                                 pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

        ## VARIABLES
        # Flow variables, the amount of flow along each edge
        # stored as a list flow_variables[ith_flow][jth_edge]
        flow_variables = []
        for i in range(len(self.commodities)):
            flow_variable_edges = []
            for j in range(len(self.edges)):
                flow_variable_edges.append(
                    solver.NumVar(0, solver.infinity(), '({},{})'.format(i, j)))
            flow_variables.append(flow_variable_edges)

        ## CONSTRAINTS
        # Flow from source constraint (net flow must equal demand)
        conservation_source_constraints = []
        for i, commodity in enumerate(self.commodities):
            # create constraint
            constraint_i = solver.Constraint(demands[i] - epsilon,
                                             demands[i] + epsilon,
                                             '(source,{})'.format(i))
            for edge_index in [self.edge_index_dict[edge] for edge in
                               self.graph.out_edges(commodity[0])]:
                # out flow is positive
                constraint_i.SetCoefficient(flow_variables[i][edge_index], 1)
            for edge_index in [self.edge_index_dict[edge] for edge in
                               self.graph.in_edges(commodity[0])]:
                # in flow is negative
                constraint_i.SetCoefficient(flow_variables[i][edge_index], -1)
            conservation_source_constraints.append(constraint_i)

        # Flow to sink constraint (in flow must equal demand, out must be zero)
        conservation_sink_constraints = []
        for i, commodity in enumerate(self.commodities):
            # create in flow constraint
            constraint_i_in = solver.Constraint(-demands[i] - epsilon,
                                                -demands[i] + epsilon,
                                                '(sink_in,{})'.format(i))
            for edge_index in [self.edge_index_dict[edge] for edge in
                               self.graph.in_edges(commodity[1])]:
                # in flow is negative
                constraint_i_in.SetCoefficient(flow_variables[i][edge_index],
                                               -1)
            conservation_sink_constraints.append(constraint_i_in)

            constraint_i_out = solver.Constraint(0, 0,
                                                 '(sink_out,{})'.format(i))
            for edge_index in [self.edge_index_dict[edge] for edge in
                               self.graph.out_edges(commodity[1])]:
                # out flow is positive
                constraint_i_out.SetCoefficient(flow_variables[i][edge_index],
                                                1)
            conservation_sink_constraints.append(constraint_i_out)

        # Flow at transit node constraint (net flow must be zero)
        conservation_transit_constraints = []
        for i, commodity in enumerate(self.commodities):
            constraints_flow_i = []
            for j in range(self.graph.number_of_nodes()):
                if j != commodity[0] and j != commodity[1]:
                    # create constraint
                    constraint_j = solver.Constraint(-epsilon, +epsilon,
                                                     '(transit,{},{})'.format(i,
                                                                              j))
                    for edge_index in [self.edge_index_dict[edge] for edge in
                                       self.graph.out_edges(j)]:
                        # out flow is positive
                        constraint_j.SetCoefficient(
                            flow_variables[i][edge_index],
                            1)
                    for edge_index in [self.edge_index_dict[edge] for edge in
                                       self.graph.in_edges(j)]:
                        # in flow is negative
                        constraint_j.SetCoefficient(
                            flow_variables[i][edge_index],
                            -1)
                    constraints_flow_i.append(constraint_j)
            conservation_transit_constraints.append(constraints_flow_i)

        # Flow splitting at transit constraints (edge flow must be correct split of
        # in flow)
        splitting_ratio_constraints = []
        for i, commodity in enumerate(self.commodities):
            constraints_flow_i = []
            for j in range(self.graph.number_of_nodes()):
                # Sink has not such constraint and we handle source differently
                if j != commodity[1] and j != commodity[0]:
                    in_edges = [self.edge_index_dict[edge] for edge in
                                self.graph.in_edges(j)]
                    out_edges = [self.edge_index_dict[edge] for edge in
                                 self.graph.out_edges(j)]

                    # separate constraint for split of each out_edge taking into
                    # account all in_edges
                    for out_edge_index in out_edges:
                        # create constraint
                        constraint_edge = \
                            solver.Constraint(-epsilon, +epsilon,
                                              '(split,{},{},{})'.format(
                                                  i, j,
                                                  out_edge_index))
                        split_ratio = routing[i][out_edge_index]
                        # flow on out edge
                        constraint_edge.SetCoefficient(
                            flow_variables[i][out_edge_index], 1)
                        for in_edge_index in in_edges:
                            # should equal sum of flow on all in edges scaled by
                            # split ratio
                            constraint_edge.SetCoefficient(
                                flow_variables[i][in_edge_index],
                                -1 * split_ratio)
                        constraints_flow_i.append(constraint_edge)
            splitting_ratio_constraints.append(constraints_flow_i)

        # Flow splitting at source constraints (edge flow must be correct split of
        # in flow + demand)
        source_splitting_constraints = []
        for i, commodity in enumerate(self.commodities):
            constraints_flow_i = []
            in_edges = [self.edge_index_dict[edge] for edge in
                        self.graph.in_edges(commodity[0])]
            out_edges = [self.edge_index_dict[edge] for edge in
                         self.graph.out_edges(commodity[0])]
            for out_edge_index in out_edges:
                # create constraint
                split_ratio = routing[i][out_edge_index]
                split_demand = split_ratio * demands[i]
                constraint_edge = \
                    solver.Constraint(split_demand - epsilon,
                                      split_demand + epsilon,
                                      '(split,{},{},{})'.format(i, j,
                                                                out_edge_index))
                # flow on out edge
                constraint_edge.SetCoefficient(
                    flow_variables[i][out_edge_index], 1)
                for in_edge_index in in_edges:
                    # should equal sum of flow on all in edges scaled by split ratio
                    constraint_edge.SetCoefficient(
                        flow_variables[i][in_edge_index],
                        -1 * split_ratio)
                constraints_flow_i.append(constraint_edge)
            source_splitting_constraints.append(constraints_flow_i)

        # ## OBJECTIVES TODO: this is currently in progress (but not at all done)
        # # Make the splitting ratios an objective to help with returning at least
        # # some solution. Also can add how much we've missed by to the reward
        # # First we add more constraints so that we are minimising the maximum
        # max_utilisation_variable = solver.NumVar(0, solver.Infinity(),
        #                                          'max_link_utilisation')
        # min_of_max_constraints = []
        # for i, edge in enumerate(self.edges):
        #     # Constraint that '-inf < f_0 + f_1 +... - max < 0'
        #     # i.e 'f_0 + f_1 + ... < max'
        #     constraint_i = solver.Constraint(-solver.Infinity(), 0,
        #                                      '(5,{})'.format(i))
        #     constraint_i.SetCoefficient(max_utilisation_variable, -1)
        #     for j, flow_variable in enumerate(flow_variables):
        #         constraint_i.SetCoefficient(flow_variable[i],
        #                                     demands[j] /
        #                                     self.graph.get_edge_data(*edge)[
        #                                         'weight'])
        #     min_of_max_constraints.append(constraint_i)
        #
        # # Objective now is to minimise the maximum link utilisation
        # objective = solver.Objective()
        # objective.SetCoefficient(max_utilisation_variable, 1)
        # objective.SetMinimization()
        solver.Solve()

        result_status = solver.Solve()

        # extract the actual routing. Useful for debugging, maybe use to bootstrap
        utilisation = np.zeros(
            (len(self.commodities), self.graph.number_of_edges()))
        assignment = np.zeros(
            (len(self.commodities), self.graph.number_of_edges()))

        # if routing is really that bad, just bail and give a sad result
        if result_status == solver.NOT_SOLVED or  result_status == solver.INFEASIBLE:
            return 1.0

        for i in range(len(self.commodities)):
            for j in range(self.graph.number_of_edges()):
                utilisation[i][j] = flow_variables[i][j].solution_value() / \
                                    self.edges[j][2]['weight']
                assignment[i][j] = flow_variables[i][j].solution_value()

        return np.max(np.sum(utilisation, axis=0))
