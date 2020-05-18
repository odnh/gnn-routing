from typing import Tuple, List, Dict, Type

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
            self.graph[edge[0]][edge[1]]['route_weight'] = (
                                                                   (action[
                                                                        i] + 1.0) / 2.0) + EPSILON

        # TODO: revisit
        # rescale softmin gamma from action range [-1, 1] to [0, e^2]
        # is not linear because action of gamma is not linear
        gamma = np.exp((action[-1] + 1.0) * 1.5) - 1.0

        # then for each flow we calculate the splitting ratios
        for flow_idx, flow in enumerate(self.flows):
            # first we get distance to dest values for each node
            distance_results = nx.single_source_bellman_ford_path_length(
                self.graph, flow[1], weight='route_weight')
            distances = np.zeros(self.graph.number_of_nodes(), dtype=np.float)

            # make the distances lookupable by node
            for (target, distance) in distance_results.items():
                distances[target] = distance

            # now we prune edges that take us further from the destination so
            # that there are no cycles
            pruned_graph = self.graph.copy()
            for (src, dst) in self.graph.edges():
                if distances[dst] >= distances[src]:
                    pruned_graph.remove_edge(src, dst)

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

    def softmin(self, array: np.ndarray, gamma: float) -> np.ndarray:
        """
        Calculates and returns the softmin of an np array
        Args:
            array: an np array
            gamma: scaling value for softmin (must be positive)
        Returns:
            np array the same size but softminned
        """
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
                   (3 * graph.number_of_edges()),),
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
        # current softmin gamma. Should be [-1, 1]
        self.gamma = 0.0

        # save the last reward so in each step we see how much we improved
        self.last_reward = 0.0
        # save the reward for the entire iteration for logging
        self.iteration_reward = 0.0

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
        self.gamma = action[-1]
        self.edge_set[edge_idx] = 1

        routing = self.get_routing(self.edge_values + [self.gamma])

        # calculate and save reward
        reward = self.get_reward(routing)
        # so reward given to learner is actually the improvement
        comparison_reward = reward - self.last_reward
        self.last_reward = reward

        # iteration start: update dm and shuffle the edge order
        #                  also calc prev routing and give reward
        if self.iter_idx == 0:
            # reset array of which edges have been set
            self.edge_set = np.zeros(self.graph.number_of_edges(), dtype=float)
            # change order in which we ask for edges to be set
            np.random.shuffle(self.edge_order)
            # Set to midvalue at start so algorithm can change each edge to be
            # more or less favourable
            self.edge_values = np.zeros(self.graph.number_of_edges(),
                                        dtype=float)
            self.gamma = 0.0
            # move forwards to next dm in this sequence
            self.dm_index += 1
            self.iteration_reward = self.last_reward  # save the reward for the iteration for debugging

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
                routing = self.get_routing(self.edge_values)
                self.last_reward = self.get_reward(routing)

        self.iter_idx = (self.iter_idx + 1) % self.iter_length

        data_dict = self.get_data_dict()
        return self.get_observation(), comparison_reward, self.done, data_dict

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
        # generalisation)
        self.edge_order = np.arange(self.graph.number_of_edges())
        np.random.shuffle(self.edge_order)
        self.iter_idx = 0
        self.edge_set = np.zeros(self.graph.number_of_edges(), dtype=float)
        # intial edge values are midvalues of the action space
        self.edge_values = np.zeros(self.graph.number_of_edges(), dtype=float)
        self.gamma = 0.0
        # initialise reward to that given by initial edge values routing
        routing = self.get_routing(self.edge_values)
        self.last_reward = self.get_reward(routing)
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

        iter_info = np.empty((self.graph.number_of_edges() * 3 + 1,),
                             dtype=float)
        iter_info[0::3] = self.edge_set  # TODO: maybe get rid of this?
        iter_info[1::3] = target_edge
        iter_info[2::3] = self.edge_values  # TODO: maybe don't do this as these are unscaled
        iter_info[-1] = self.gamma
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
                          'gamma': self.gamma,
                          'real_reward': self.iteration_reward})
        # Extra logging for tensorboard at the end of an episode
        if self.done:
            data_dict.update(
                {'episode': {'r': self.iteration_reward,
                             'l': self.dm_index}})
            self.episode_total_reward = 0.0
        return data_dict
