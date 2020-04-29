from typing import Tuple, List, Dict, Type

import gym
import networkx as nx
import numpy as np
from scipy.special import softmax

from .max_link_utilisation import MaxLinkUtilisation

Routing = np.ndarray
Demand = Tuple[np.ndarray, float]
Observation = np.ndarray
Action = np.ndarray


class DDREnv(gym.Env):
    """
    Gym env for data driven routing

    Observations are: last k routing and DMs
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
            shape=(dm_memory_length * graph.number_of_nodes() *
                   (graph.number_of_nodes() - 1),),
            dtype=np.float64)

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
            # Move to the next dm sequence
            self.dm_sequence_index = (self.dm_sequence_index + 1) % len(
                self.dm_sequence)
            return self.get_observation(), 0.0, self.done, dict()
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
        self.dm_index = 0
        self.dm_memory = [self.dm_sequence[self.dm_sequence_index][i] for i in
                          range(self.dm_memory_length)]
        self.dm_index += self.dm_memory_length
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
        NB: this should never really be used.
        """
        num_edges = self.graph.number_of_edges()
        num_demands = self.graph.number_of_nodes() * \
                      (self.graph.number_of_nodes() - 1)
        return action.reshape((num_demands, num_edges))

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
        Flattens observation for input to learners
        Returns:
            A flat np array of the demand matrix
        """
        dm_no_opt_memory = [dm[0] for dm in self.dm_memory]
        return np.stack(dm_no_opt_memory).ravel()

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

        # For calculating the action space size and routing translation
        # sorted to match encoding ordering of nodes
        self.out_edge_count = [i[1] for i in
                               sorted(graph.out_degree(), key=lambda x: x[0])]

        self.action_space = gym.spaces.Box(
            low=1.0,
            high=1.0,
            shape=(sum(self.out_edge_count) * (graph.number_of_nodes() - 1),),
            dtype=np.float64)

        # Precompute list of flows for use in routing translation
        # Ordering is eg 0,1 0,2 0,3 1,0 1,2 1,3 2,0 2,1 2,3
        num_nodes = self.graph.number_of_nodes()
        self.flows = [(i, j) for i in range(num_nodes) for j in range(num_nodes)
                      if i != j]

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
            action: flat 1x(|V|*(|V|-1)*out_edges)
        Returns:
            A fully specified routing (dims 0: flows, 1: edges)
        NB: this code is currently incorrect. Do not use.
        """

        num_edges = self.graph.number_of_edges()
        num_nodes = self.graph.number_of_nodes()
        softmaxed_routing = []
        idx = 0
        for i, count in enumerate(self.out_edge_count):
            vertex = []
            for dest in range(num_nodes - 1):
                dest = []
                for _ in range(count):
                    dest.append(action[idx])
                    idx += 1
                vertex.append(softmax(dest))
            softmaxed_routing.append(vertex)

        full_routing = np.zeros((len(self.flows), num_edges), dtype=np.float32)

        for i, (_, dst) in enumerate(self.flows):
            # Ensure that edges are sorted for correct order in action
            for j, edge in enumerate(sorted(self.graph.edges())):
                relative_dst = dst
                if dst > edge[0]:
                    relative_dst -= 1
                if dst == edge[0]:
                    full_routing[i][j] = 0.0
                else:
                    full_routing[i][j] = softmaxed_routing[
                        edge[0]][relative_dst][self.edge_index[edge]]

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
                 gamma: float = 2, **kwargs):
        super().__init__(dm_sequence, dm_memory_length, graph, **kwargs)

        # Precompute list of flows for use in routing translation
        num_nodes = self.graph.number_of_nodes()
        self.flows = [(i, j) for i in range(num_nodes) for j in range(num_nodes)
                      if i != j]
        self.gamma = gamma

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(graph.number_of_edges(),),
            dtype=np.float64)

        # Indices of the edges for lookup in routing translation
        self.edge_index_map = {edge: i for i, edge in
                               enumerate(sorted(self.graph.edges()))}

    def get_routing(self, action: Action) -> Routing:
        """
        Converts a softmin routing to full routing
        Args:
            action: dims 0: edges
        Returns:
            A fully specified routing (dims 0: flows, 1: edges)
        """
        # TODO: optimise this method. Is too slow
        full_routing = np.zeros((len(self.flows), self.graph.number_of_edges()),
                                dtype=np.float32)

        # First we place the routing edge weights on the graph (and rescale
        # between 1 and 0)
        for i, edge in enumerate(sorted(self.graph.edges)):
            self.graph[edge[0]][edge[1]]['route_weight'] = (
                    (action[i] + 1.0) / 2.0)

        # then for each flow we calculate the splitting ratios
        for flow_idx, flow in enumerate(self.flows):
            # first we get distance to dest values for each node
            distance_results = nx.single_source_bellman_ford_path_length(
                self.graph, flow[1], weight='route_weight')
            distances = np.zeros(self.graph.number_of_nodes(), dtype=np.float)
            for (target, distance) in distance_results.items():
                distances[target] = distance
            # then we calculate softmin splitting for the out-edges on each node
            for node in range(self.graph.number_of_nodes()):
                out_edges = list(self.graph.out_edges(node))
                out_edge_weights = np.zeros(len(out_edges))
                # collect the weights to use for deciding splitting ratios
                # from this node
                for out_edge_idx, out_edge in enumerate(out_edges):
                    out_edge_weights[out_edge_idx] = \
                        self.graph[out_edge[0]][out_edge[1]]['route_weight'] + \
                        distances[out_edge[1]]
                # softmin the out_edge weights so that ratios sum to one
                softmin_weights = self.softmin(out_edge_weights)
                # assign to the splitting ratios for this node and flow to
                # overall routing
                for out_edge_idx, weight in enumerate(softmin_weights):
                    full_routing[flow_idx][
                        self.edge_index_map[out_edges[out_edge_idx]]] = weight

        return full_routing

    def softmin(self, array: np.ndarray) -> np.ndarray:
        """
        Calculates and returns the softmin of an np array
        Args:
            array: an np array
        Returns:
            np array the same size but softminned
        """
        exponentiated = np.exp(np.multiply(array, -self.gamma))
        total = sum(exponentiated)
        return np.divide(exponentiated, total)


class DDREnvIterative(DDREnvSoftmin):
    """
    DDREnv where routing at each edge is performed iteratively to allow for
    generalisation after learning. i.e. each step is to only get the value for
    one edge which is selected in the observation at the start of the step.
    """

    # TODO: try non-softmin version too (i.e. dest-based)

    def __init__(self,
                 dm_sequence: List[List[Demand]],
                 dm_memory_length: int,
                 graph: nx.DiGraph,
                 **kwargs):
        super().__init__(dm_sequence, dm_memory_length, graph, **kwargs)

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float64)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=np.inf,
            # Space contains per edge to set and already set
            shape=(dm_memory_length * graph.number_of_nodes() *
                   (graph.number_of_nodes() - 1) +
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

        # save the last reward so in each step we see how much we improved
        self.last_reward = 0

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
        edge_idx = self.edge_order[(self.iter_idx - 1) % self.iter_length]
        # shift from -1->1 to 0->1
        self.edge_values[edge_idx] = (action[0] + 1.0) / 2.0
        self.edge_set[edge_idx] = 1

        routing = self.get_routing(self.edge_values)

        # calculate and save reward
        reward = self.get_reward(routing)
        # so reward given to learner is actually the improvement
        comparison_reward = reward - self.last_reward
        self.last_reward = reward

        # iteration start: update dm and shuffle the edge order
        #                  also calc prev routing and give reward
        # TODO: see how it performs without the shuffle
        if self.iter_idx == 0:
            self.edge_set = np.zeros(self.graph.number_of_edges(), dtype=float)
            # Set to midvalue at start so algorithm can change each edge to be
            # more or less favourable
            self.edge_values = np.full(self.graph.number_of_edges(),
                                       0.5,
                                       dtype=float)
            self.dm_index += 1
            if self.dm_index == len(self.dm_sequence[self.dm_sequence_index]):
                self.done = True
                self.dm_sequence_index = (self.dm_sequence_index + 1) % len(
                    self.dm_sequence)
            else:
                new_dm = self.dm_sequence[self.dm_sequence_index][self.dm_index]
                self.dm_memory.append(new_dm)
                if len(self.dm_memory) > self.dm_memory_length:
                    self.dm_memory.pop(0)
            np.random.shuffle(self.edge_order)
        # inside an iteration so just change the index
        self.iter_idx = (self.iter_idx + 1) % self.iter_length

        data_dict = self.get_data_dict()

        return self.get_observation(), comparison_reward, self.done, data_dict

    def reset(self) -> Observation:
        self.dm_index = 0
        self.dm_memory = [self.dm_sequence[self.dm_sequence_index][i] for i in
                          range(self.dm_memory_length)]
        self.dm_index += self.dm_memory_length
        self.done = False

        # iteration variables
        self.iter_length = self.graph.number_of_edges()
        self.edge_order = np.arange(self.graph.number_of_edges())
        np.random.shuffle(self.edge_order)
        self.iter_idx = 0
        self.edge_set = np.zeros(self.graph.number_of_edges(), dtype=float)
        self.edge_values = np.zeros(self.graph.number_of_edges(), dtype=float)
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

        iter_info = np.empty((self.graph.number_of_edges() * 2,), dtype=float)
        iter_info[0::2] = self.edge_set
        iter_info[1::2] = target_edge
        dm_no_opt_memory = [dm[0] for dm in self.dm_memory]
        return np.concatenate((np.stack(dm_no_opt_memory).ravel(), iter_info))

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
                          'real_reward': self.last_reward})
        # Extra logging for tensorboard at the end of an episode
        if self.done:
            data_dict.update(
                {'episode': {'r': self.last_reward,
                             'l': self.dm_index}})
            self.episode_total_reward = 0.0
        return data_dict
