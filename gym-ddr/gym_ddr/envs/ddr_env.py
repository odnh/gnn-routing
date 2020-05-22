import collections
from queue import PriorityQueue
from typing import Tuple, List, Dict, Type, Set

import gym
import networkx as nx
import numpy as np

from .max_link_utilisation import MaxLinkUtilisation

Routing = np.ndarray
Demand = Tuple[np.ndarray, float]
Observation = np.ndarray
Action = np.ndarray

EPSILON = 1e-5


def normalise_array(array: np.ndarray) -> np.ndarray:
    """
    Takes an array of positive values and normalises between 1 and 0
    Args:
        array: np array

    Returns:
        normalised np array
    """
    return np.divide(array, np.sum(array)) if np.sum(array) != 0 else array


class DDREnv(gym.Env):
    """
    Gym env for data driven routing

    Observations are: sum of in and out demand for each node
    Actions are: a routing

    Actions are fully specified routing (i.e. splitting ratio for each flow over
    each edge). Subclass to either take a less specified version and transform
    otherwise learner will need to change too
    Rewards are: utilisation under routing compared to maximum utilisation
    """

    def __init__(self,
                 dm_sequence: List[List[Demand]],
                 dm_memory_length: int,
                 graph: nx.DiGraph,
                 oblivious_routing: np.ndarray = None):
        """
        Args:
          dm_sequence: the sequence of sequences of dms to use. This is a list
                       of dm sequences where each dm in the sequence is a tuple
                       of demand matrix and optimal max-link-utilisation
          dm_memory_length: the length of the dm history we should train on
          graph: the graph we will be routing over
          oblivious_routing: an oblivious routing
        """
        self.dm_sequence = dm_sequence
        self.dm_index = 0  # index of the dm within a sequence we are on
        self.dm_sequence_index = 0  # index of the sequence we are on
        self.dm_memory_length = dm_memory_length
        self.dm_memory: List[Demand] = []
        self.graph = graph
        self.done = False

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(graph.number_of_nodes() * (graph.number_of_nodes() - 1) *
                   graph.number_of_edges(),),
            dtype=np.float64)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=np.inf,
            shape=(dm_memory_length * graph.number_of_nodes() * 2,),
            dtype=np.float64)

        # so that we can make the input routing valid
        self.out_edge_count = np.cumsum([i[1] for i in
                                         sorted(graph.out_degree(),
                                                key=lambda x: x[0])])

        # Precompute list of flows for use in routing and dm translation
        num_nodes = self.graph.number_of_nodes()
        self.flows = [(i, j) for i in range(num_nodes) for j in range(num_nodes)
                      if i != j]

        self.oblivious_routing = oblivious_routing

        self.mlu = MaxLinkUtilisation(graph)
        self.opt_utilisation = 0.0
        self.utilisation = 0.0

    def step(self, action: np.ndarray) -> Tuple[Observation,
                                                float,
                                                bool, Dict[None, None]]:
        """
        Args:
          action: a routing this is a fully specified routing (must be 1D
                  ndarray)
        Returns:
          history of dms and the other bits and pieces expected (use np.stack
          on the history for training)
        """
        # Check if sequence is exhausted
        if self.done:
            return self.get_observation(), 0.0, self.done, dict()

        # update dm and history
        self.dm_index += 1
        if self.dm_index == len(self.dm_sequence[self.dm_sequence_index]):
            self.done = True
            reward = 0.0  # TODO: maybe need more sensible reward when set done
            data_dict = {}
        else:
            new_dm = self.dm_sequence[self.dm_sequence_index][self.dm_index]
            self.dm_memory.append(new_dm)
            if len(self.dm_memory) > self.dm_memory_length:
                self.dm_memory.pop(0)
            routing = self.get_routing(action)
            reward = self.get_reward(routing)
            data_dict = self.get_data_dict()
        return self.get_observation(), reward, self.done, data_dict

    def reset(self) -> Observation:
        # first choose a random dm sequence
        self.dm_sequence_index = np.random.randint(0, len(self.dm_sequence))
        # tne place us at the start of that sequence
        self.dm_index = 0
        # pre-fill the memory from the sequence (backwards to queue works
        # properly)
        self.dm_memory = [self.dm_sequence[self.dm_sequence_index][i] for i in
                          range(self.dm_memory_length-1, -1, -1)]
        # move the index to the first element after the last element of memory
        # (this is rather than the next as is incremented at the start of each
        # step
        self.dm_index = self.dm_memory_length
        self.done = False
        return self.get_observation()

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def get_routing(self, action: Action) -> Routing:
        """
        Subclass to use different actions, assumes action is a fully specified
        routing in base case but also does any necessary unflattening.
        """
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        num_demands = self.graph.number_of_nodes() * \
                      (self.graph.number_of_nodes() - 1)

        # we are required to make the destination routing valid using softmax
        # over out_edges from each node
        normalised_routing = np.zeros((num_demands, num_edges))
        reshaped_action = np.reshape(action, (num_demands, num_edges))
        rescaled_action = np.divide(np.add(reshaped_action, 1.0), 2.0)
        for flow_idx in range(num_demands):
            for intermediate_node in range(num_nodes):
                start_idx = self.out_edge_count[
                    intermediate_node - 1] if intermediate_node - 1 >= 0 else 0
                end_idx = self.out_edge_count[intermediate_node]
                normalised_routing[flow_idx][
                start_idx:end_idx] = normalise_array(
                    rescaled_action[flow_idx][start_idx:end_idx])

        return normalised_routing

    def get_reward(self, routing: Routing) -> float:
        """
        Reward calculated as utilisation of graph given routing compared to
        optimal. May have to call external libraries to calculate efficiently.
        """
        self.utilisation = self.mlu.calc(self.dm_memory[0][0], routing)
        self.opt_utilisation = self.dm_memory[0][1]
        return -(self.utilisation / self.opt_utilisation)

    def get_observation(self) -> Observation:
        """
        Takes the dms and presents a history of outgoing and incoming demands
        per node (flattened)
        Returns:
            A flat np array of the demands
        """
        node_demands_memory = []
        for dm in self.dm_memory:
            node_demands = np.zeros(self.graph.number_of_nodes() * 2,
                                    dtype=float)
            for flow_idx, (src, dst) in enumerate(self.flows):
                node_demands[src * 2] += dm[0][flow_idx]
                node_demands[(dst * 2) + 1] += dm[0][flow_idx]
            node_demands_memory.append(node_demands)
        # normalise observation into [0,1]
        normalised_node_demands_memory = [normalise_array(node_demands) for
                                          node_demands in node_demands_memory]
        observation = np.stack(normalised_node_demands_memory).ravel()
        return observation

    def get_data_dict(self) -> Dict:
        """
        Gets a data dict to return in the step function. Contains the
        utilisation under oblivious routing
        Returns:
            dict with utilisation under oblivious routing, optimal and action
        """
        data_dict = {
            'utilisation': self.utilisation,
            'opt_utilisation': self.opt_utilisation
        }
        if self.oblivious_routing is not None:
            data_dict['oblivious_utilisation'] = self.mlu.calc(
                self.dm_memory[0][0], self.oblivious_routing)
        return data_dict


class DDREnvDest(DDREnv):
    """
    DDR Env where routing is destination-based which significantly reduces
    the action space. This class simply performs the translation to a full
    routing.
    """

    def __init__(self,
                 dm_sequence: List[List[Demand]],
                 dm_memory_length: int,
                 graph: nx.DiGraph, **kwargs):
        super().__init__(dm_sequence, dm_memory_length, graph, **kwargs)

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(
                self.graph.number_of_nodes() * self.graph.number_of_edges(),),
            dtype=np.float64)

        # Indices of the edges for lookup in routing translation
        # Takes each edge of graph to an index under its source node
        self.edge_index = {}
        for node in range(graph.number_of_nodes()):
            count = 0
            for edge in sorted(graph.out_edges(node)):
                self.edge_index[edge] = count
                count += 1

    def get_routing(self, action: Action) -> Routing:
        """
        Converts a destination routing to full routing
        Args:
            action: flat 1x(|V|*|E|)
        Returns:
            A fully specified routing (dims 0: flows, 1: edges)
        """
        # first we make the destination routing valid using softmax over
        # out_edges from each node
        num_edges = self.graph.number_of_edges()
        num_nodes = self.graph.number_of_nodes()
        normalised_routing = np.zeros((num_nodes, num_edges))
        reshaped_action = np.reshape(action, (num_nodes, num_edges))
        # move from [-1,1] to [0, 1]
        rescaled_action = np.divide(np.add(reshaped_action, 1.0), 2.0)
        for flow_dest in range(num_nodes):
            for intermediate_node in range(num_nodes):
                start_idx = self.out_edge_count[
                    intermediate_node - 1] if intermediate_node - 1 >= 0 else 0
                end_idx = self.out_edge_count[intermediate_node]
                normalised_routing[flow_dest][
                start_idx:end_idx] = normalise_array(
                    rescaled_action[flow_dest][start_idx:end_idx])

        # then we assign it identically over all sources
        full_routing = np.zeros((len(self.flows), num_edges), dtype=np.float32)
        for flow_idx, (_, dst) in enumerate(self.flows):
            full_routing[flow_idx] = normalised_routing[dst]

        return full_routing


class DDREnvSoftmin(DDREnv):
    """
    DDR Env where softmin routing is used (from Learning to Route with
    Deep RL paper). Routing is a single weight per edge, transformed to
    splitting ratios for input to the optimizer calculation.
    """

    def __init__(self,
                 dm_sequence: List[List[Demand]],
                 dm_memory_length: int,
                 graph: nx.DiGraph,
                 gamma: float = 2.0, **kwargs):
        super().__init__(dm_sequence, dm_memory_length, graph, **kwargs)

        # Precompute list of flows for use in routing translation
        num_nodes = self.graph.number_of_nodes()
        self.flows = [(i, j) for i in range(num_nodes) for j in range(num_nodes)
                      if i != j]
        self.gamma_fixed = gamma

        # plus one is the softmin_gamma
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(graph.number_of_edges() + 1,),
            dtype=np.float64)

        # Indices of the edges for lookup in routing translation
        self.edge_index_map = {edge: i for i, edge in
                               enumerate(sorted(self.graph.edges()))}

    def get_routing(self, action: Action) -> Routing:
        """
        Converts a softmin routing to full routing and prunes cycles with a
        very basic heuristic
        Args:
            action: dims 0: edges, final value is gamma
        Returns:
            A fully specified routing (dims 0: flows, 1: edges)
        """
        full_routing = np.zeros((len(self.flows), self.graph.number_of_edges()),
                                dtype=np.float32)

        # TODO: revisit
        # First we place the routing edge weights on the graph (and rescale
        # between 0 and 1 but plus epsilon so that no edges have 0 weight which
        # causes pruning issues when removing cycles)
        for i, edge in enumerate(sorted(self.graph.edges)):
            self.graph[edge[0]][edge[1]]['route_weight'] = ((action[i] + 1.0) / 2.0) + EPSILON

        # TODO: revisit
        # rescale softmin gamma from action range [-1, 1] to [0, e^2]
        # is not linear because action of gamma is not linear
        gamma = np.exp((action[-1] + 1.0) * 1.5) - 1.0

        # then for each flow we calculate the splitting ratios
        for flow_idx, flow in enumerate(self.flows):
            # first we prune the graph down to a DAG
            pruned_graph = self.prune_graph(self.graph, flow)

            # then we get distance to dest values for each node
            distances = collections.defaultdict(int)
            distance_results = nx.shortest_path_length(pruned_graph, source=None, target=flow[1], weight='route_weight')
            distances.update(distance_results)

            # then we calculate softmin splitting for the out-edges on each node
            for node in range(self.graph.number_of_nodes()):
                out_edges = list(pruned_graph.out_edges(node))
                out_edge_weights = np.zeros(len(out_edges))
                # collect the weights to use for deciding splitting ratios
                # from this node
                for out_edge_idx, out_edge in enumerate(out_edges):
                    out_edge_weights[out_edge_idx] = \
                        pruned_graph[out_edge[0]][out_edge[1]]['route_weight'] + \
                        distances[out_edge[1]]
                # softmin the out_edge weights so that ratios sum to one
                softmin_weights = self.softmin(out_edge_weights, gamma)
                # assign to the splitting ratios for this node and flow to
                # overall routing
                for out_edge_idx, weight in enumerate(softmin_weights):
                    full_routing[flow_idx][
                        self.edge_index_map[out_edges[out_edge_idx]]] = weight
        return full_routing

    def prune_graph_simple(self, graph: nx.DiGraph, flow: Tuple[int, int]) -> nx.DiGraph:
        """
        Remove cycles between flow source and sink. Uses distances to give a
        partial topological order then removes edges that take us in the wrong
        direction. Simple but removes more paths than necessary.
        Args:
            graph: graph to DAGify
            flow: source and sink of the flow

        Returns:
            A DAG with source at the start and sink at the end
        """
        graph = graph.copy()

        # first calculate distance to sink for each vertex
        distances = collections.defaultdict(int)
        distance_results = nx.shortest_path_length(graph, source=None,
                                                   target=flow[1],
                                                   weight='route_weight')
        distances.update(distance_results)

        # now we prune edges that take us further from the destination so
        # that there are no cycles
        for (src, dst) in list(graph.edges()):
            if distances[dst] >= distances[src]:
                graph.remove_edge(src, dst)
        return graph

    def prune_graph(self, graph: nx.DiGraph, flow: Tuple[int, int]) -> nx.DiGraph:
        """
        Makes the graph a DAG, retaining as many paths as possible (although not
        mathematically) with flow source being the only parentless node and dst
        being the only sink.
        Args:
            graph: Graph with edge weights to prune edges
            flow: A pair of nodes to be source and destination

        Returns:
            A DAG from source to destination
        """
        graph = graph.copy()
        to_explore: PriorityQueue[int] = PriorityQueue()
        to_explore.put((0, flow[0], []))
        # maps node to its parent. Nodes must have at most one parent unless
        # they are "on_path"
        parents_map: Dict[List[int]] = collections.defaultdict(list)
        # list of edges where our frontier butts up against itself. We need to
        # carefully prune edges around this point to remove cycles
        frontier_meets: Set[Tuple[int, int]] = set()

        # first we explore all the nodes from the source
        explored_nodes: Set[int] = set()
        while not to_explore.empty():
            distance, current_node, parents = to_explore.get()
            # see if we've already been to this node
            if current_node in explored_nodes:
                continue

            # set our parent(s)
            parents_map[current_node] = parents

            # get the neighbours but remove the one we got here from
            neighbours = set(graph.neighbors(current_node))
            neighbours.difference_update(parents)

            # get ready to explore the neighbours
            for neighbour in neighbours:
                if neighbour == flow[1]:
                    parents_map[flow[1]].append(current_node)
                elif neighbour in explored_nodes:
                    smallest = min(current_node, neighbour)
                    largest = max(current_node, neighbour)
                    frontier_meets.add((smallest, largest))
                else:
                    # put the neighbour on the queue of nodes to explore
                    to_explore.put((distance + graph[current_node][neighbour]['route_weight'], neighbour, [current_node]))

            # we've explored this node so add it to the list
            explored_nodes.add(current_node)

        # now we traceback from the dst to see which nodes are on the right path
        to_explore_trace: List[int] = [flow[1]]
        on_path = set()
        dest_dist = {flow[1]: 0}
        while to_explore_trace:
            current_node = to_explore_trace.pop(0)
            # see if we've already been here
            if current_node in on_path:
                continue

            # get ready to trace back to the parents
            for parent in parents_map[current_node]:
                to_explore_trace.append(parent)
                dest_dist[parent] = dest_dist[current_node] + graph[parent][current_node]['route_weight']
            # remember that his node is on the path src to dst
            on_path.add(current_node)

        # now we add frontier meets to the path
        for node_a, node_b in frontier_meets:
            # find the distance from dst of first ancestor that is on_path for
            # each node
            ancestor_on_path_a = self.trace_to_on_path(node_a, parents_map, on_path)
            ancestor_on_path_b = self.trace_to_on_path(node_b, parents_map, on_path)
            if dest_dist[ancestor_on_path_a] > dest_dist[ancestor_on_path_b]:
                path_start = node_a
                path_end = node_b
                ancestor_end = ancestor_on_path_b
            elif dest_dist[ancestor_on_path_b] > dest_dist[ancestor_on_path_a]:
                path_start = node_b
                path_end = node_a
                ancestor_end = ancestor_on_path_a
            else:
                # this could lead to a loops so don't use this path
                continue

            # TODO: try some sort of distance interpolation instead maybe?
            path_dist = dest_dist[ancestor_end]

            ## we want to direct flow the other way along here, so reparent, set
            # on_path and give a dest_dist
            current = path_end
            previous = path_start
            while current not in on_path:
                # put on_path
                on_path.add(current)
                # give a dest_dist
                dest_dist[current] = path_dist
                # get next node on the path
                parent = parents_map[current][0]
                # flip our parent pointer
                parents_map[current] = [previous]
                # now we hop up the path
                previous = current
                current = parent
            # reparent point where this path meets on_path
            parents_map[ancestor_end].append(previous)

            # finally we give all the path_start nodes a correct dest_dist and
            # set on_path
            current = path_start
            while current not in on_path:
                dest_dist[current] = path_dist
                on_path.add(current)
                parent = parents_map[current][0]
                current = parent

        # finally, we prune links we don't need
        edges = list(graph.edges)
        for (src, dst) in edges:
            # remove edges not on the path
            if src not in on_path or dst not in on_path:
                graph.remove_edge(src, dst)
            # remove edges against the path
            elif src not in parents_map[dst]:
                graph.remove_edge(src, dst)

        return graph

    def trace_to_on_path(self, node: int, parents_map: Dict, on_path: [int]) -> int:
        """
        Part of graph pruning. Returns the first ancestor in the 'on_path' set
        Args:
            node: node to start at
            parents_map: map from nodes to lists of their parent nodes
            on_path: set of nodes that are on a path src to dst

        Returns:
            The first ancestor in on_path
        """
        current = node
        while current not in on_path:
            current = parents_map[current][0]
        return current


    def softmin(self, array: np.ndarray, gamma: float) -> np.ndarray:
        """
        Calculates and returns the softmin of an np array
        Args:
            array: an np array
            gamma: scaling value for softmin (must be positive)
        Returns:
            np array the same size but softminned
        """
        gamma = 2.0
        exponentiated = np.exp(np.multiply(array, -gamma))
        total = sum(exponentiated)
        return np.divide(exponentiated, total)


class DDREnvIterative(DDREnvSoftmin):
    """
    DDREnv where routing at each edge is performed iteratively to allow for
    generalisation after learning. i.e. each step is to only get the value for
    one edge which is selected in the observation at the start of the step.
    """

    def __init__(self,
                 dm_sequence: List[List[Demand]],
                 dm_memory_length: int,
                 graph: nx.DiGraph,
                 **kwargs):
        super().__init__(dm_sequence, dm_memory_length, graph, **kwargs)

        # action space is [edge_value, softmin_gamma]
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float64)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=np.inf,
            # Space contains per edge to set and already set
            shape=(dm_memory_length * graph.number_of_nodes() * 2 +
                   (2 * graph.number_of_edges()),),
            dtype=np.float64)

        # Extra state to track partial completeness of single routing selection
        # steps in one iteration
        self.iter_length = graph.number_of_edges()
        # the order to set edges during an iteration
        self.edge_order = np.arange(graph.number_of_edges())
        np.random.shuffle(self.edge_order)
        # index into the iteration / which edge to set
        self.iter_idx = 0
        # array of which edges have had their routing set so far
        self.edge_set = np.zeros(graph.number_of_edges(), dtype=float)
        # current softmin edge values
        self.edge_values = np.zeros(graph.number_of_edges(), dtype=float)

        # save the reward for the entire iteration for logging
        self.iteration_reward = 0.0
        self.last_gamma = 0.0

    def step(self, action: Type[np.ndarray]) -> Tuple[Observation,
                                                      float,
                                                      bool, Dict[None, None]]:
        """
        Args:
          action: a single value assignment for the edge asked to assign in the
                  observation given from the previous call
        Returns:
          history of dms and the other bits and pieces expected (use np.stack
          on the history for training)
        """
        # Check if sequence is exhausted
        if self.done:
            return self.get_observation(), 0.0, self.done, dict()

        # add action to the overall routing
        edge_idx = self.edge_order[self.iter_idx % self.iter_length]
        self.edge_values[edge_idx] = action[0]
        self.edge_set[edge_idx] = 1

        # only give reward at end of a set of iterations
        reward = 0

        self.iter_idx = (self.iter_idx + 1) % self.iter_length
        # iteration start: update dm and shuffle the edge order
        #                  also calc prev routing and give reward
        if self.iter_idx == 0:
            # calculate reward for iteration seeing as all edges have been set
            gamma = action[1]
            routing = self.get_routing(np.append(self.edge_values, gamma))
            reward = self.get_reward(routing)
            self.last_gamma = gamma  # save last used gamma for debugging
            self.iteration_reward = reward  # save the reward for the iteration for debugging

            # reset array of which edges have been set (to all zero)
            self.edge_set = np.zeros(self.graph.number_of_edges(), dtype=float)
            # change order in which we ask for edges to be set TODO: try not shuffling
            np.random.shuffle(self.edge_order)
            # zero the routing values (i.e. set to midvalue)
            self.edge_values = np.zeros(self.graph.number_of_edges(),
                                        dtype=float)

            # move forwards to next dm in this sequence
            self.dm_index += 1

            if self.dm_index == len(self.dm_sequence[self.dm_sequence_index]):
                # at end of dm sequence so end episode
                self.done = True
            else:
                # still in dm sequence so update the memory by appending new dm
                new_dm = self.dm_sequence[self.dm_sequence_index][self.dm_index]
                self.dm_memory.append(new_dm)
                if len(self.dm_memory) > self.dm_memory_length:
                    self.dm_memory.pop(0)
                # initialise reward to that given by initial edge values routing

        data_dict = self.get_data_dict()
        observation = self.get_observation()
        return observation, reward, self.done, data_dict

    def reset(self) -> Observation:
        # first choose a random dm sequence
        self.dm_sequence_index = np.random.randint(0, len(self.dm_sequence))
        # then place us at the start of that sequence
        self.dm_index = 0
        # pre-fill the memory from the sequence
        self.dm_memory = [self.dm_sequence[self.dm_sequence_index][i] for i in
                          range(self.dm_memory_length-1, -1, -1)]
        # move the index to the first element after the memory
        self.dm_index += self.dm_memory_length
        self.done = False

        ## iteration variables
        self.iter_length = self.graph.number_of_edges()
        # the order in which we want to set the edges (it is shuffled for
        # generalisation) TODO: try not shuffling
        self.edge_order = np.arange(self.graph.number_of_edges())
        np.random.shuffle(self.edge_order)
        self.iter_idx = 0
        self.edge_set = np.zeros(self.graph.number_of_edges(), dtype=float)
        # intial edge values are midvalues of the action space
        self.edge_values = np.zeros(self.graph.number_of_edges(), dtype=float)
        # initialise reward to that given by initial edge values routing
        return self.get_observation()

    def get_observation(self) -> Observation:
        """
        Flattens observation for input to learners
        Returns:
            A flat np array of the demand matrix
        """
        target_edge_idx = self.edge_order[self.iter_idx]
        target_edge = np.identity(self.graph.number_of_edges())[
                      target_edge_idx:target_edge_idx + 1]

        iter_info = np.empty((self.graph.number_of_edges() * 2,),
                             dtype=float)
        iter_info[0::2] = self.edge_set  # TODO: maybe get rid of this?
        iter_info[1::2] = target_edge
        # iter_info[2::3] = self.edge_values  # TODO: maybe don't do this as these are unscaled
        demands_history = DDREnv.get_observation(self)
        return np.concatenate((demands_history, iter_info))

    def get_data_dict(self) -> Dict:
        """
        Gets a data dict to return in the step function.
        Returns:
            dict with oblivious routing from parent and information about
            iteration state
        """
        target_edge_idx = self.edge_order[self.iter_idx]
        target_edge = np.identity(self.graph.number_of_edges())[
                      target_edge_idx:target_edge_idx + 1]
        data_dict = super().get_data_dict()
        data_dict.update({'iter_idx': self.iter_idx, 'target_edge': target_edge,
                          'edge_set': self.edge_set,
                          'values': self.edge_values,
                          'gamma': self.last_gamma,
                          'real_reward': self.iteration_reward})
        # Extra logging for tensorboard at the end of an episode
        if self.done:
            data_dict.update(
                {'episode': {'r': self.iteration_reward,
                             'l': self.dm_index}})
        return data_dict
