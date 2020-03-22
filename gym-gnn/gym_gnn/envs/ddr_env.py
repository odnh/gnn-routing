from typing import Tuple, List, Dict, Callable, Generator, Type
import gym
import networkx as nx
import numpy as np
from scipy.special import softmax

from . import max_link_utilisation

Routing = Type[np.ndarray]
Demand = Type[np.ndarray]
Observation = Type[np.ndarray]
Action = Type[np.ndarray]


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
                 dm_generator_getter: Callable[
                     [],
                     Generator[Demand, None, None]],
                 dm_memory_length: int,
                 graph: nx.DiGraph):
        """
        Args:
          dm_generator_getter: a function that returns a generator for demand
                               matrices (so can reset)
          dm_memory_length: the length of the dm history we should train on
          graph: the graph we will be routing over
        """
        self.dm_generator_getter = dm_generator_getter
        self.dm_generator = dm_generator_getter()
        self.dm_memory_length = dm_memory_length
        self.dm_memory: List[Demand] = []
        self.graph = graph
        self.done = False

        self.action_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(graph.number_of_nodes() * (graph.number_of_nodes() - 1) *
                   graph.number_of_edges()))
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(dm_memory_length * graph.number_of_nodes() *
                   (graph.number_of_nodes() - 1),))

    def step(self, action: Type[np.ndarray]) -> Tuple[Observation,
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
        new_dm = next(self.dm_generator, None)
        if new_dm is None:
            self.done = True
            return self.get_observation(), 0.0, self.done, dict()
        else:
            self.dm_memory.append(new_dm)
            if len(self.dm_memory) > self.dm_memory_length:
                self.dm_memory.pop(0)
            routing = self.get_routing(action)
            reward = self.get_reward(routing)
        return self.get_observation(), reward, self.done, dict()

    def reset(self) -> Observation:
        self.dm_generator = self.dm_generator_getter()
        self.dm_memory = [next(self.dm_generator) for _ in
                          range(self.dm_memory_length)]
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
        num_edges = self.graph.number_of_edges()
        num_demands = self.graph.number_of_nodes() *\
                      (self.graph.number_of_nodes() - 1)
        return action.reshape((num_demands, num_edges))

    def get_reward(self, routing: Routing) -> float:
        """
        Reward calculated as utilisation of graph given routing compared to
        optimal. May have to call external libraries to calculate efficiently.
        """
        utilisation = max_link_utilisation.calc(self.graph, self.dm_memory[0],
                                                routing)
        opt_utilisation = max_link_utilisation.opt(self.graph,
                                                   self.dm_memory[0])
        return -(utilisation / opt_utilisation)

    def get_observation(self) -> Observation:
        """
        Flattens observation for input to learners
        Returns: A flat np array of the demand matrix
        """
        return np.stack(self.dm_memory).ravel()


class DDREnvDest(DDREnv):
    """
    DDR Env where routing is destination-based which significantly reduces
    the action space. This class simply performs the translation to a full
    routing.
    """

    def __init__(self,
                 dm_generator_getter: Callable[
                     [],
                     Generator[Demand, None, None]],
                 dm_memory_length: int,
                 graph: nx.DiGraph):
        super().__init__(dm_generator_getter, dm_memory_length, graph)

        # For calculating the action space size and routing translation
        self.out_edge_count = [i[1] for i in graph.out_degree()]

        self.action_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(sum(self.out_edge_count) * (graph.number_of_nodes()-1)))

        # Precompute list of flows for use in routing translation
        num_nodes = self.graph.number_of_nodes()
        self.flows = [(i, j) for i in range(num_nodes) for j in range(num_nodes)
                      if i != j]

        # Indices of the edges for lookup in routing translation
        # Takes each edge of graph to an index under its source node
        self.edge_index = {}
        for node in range(graph.number_of_nodes()):
            count = 0
            for edge in graph.out_edges(node):
                self.edge_index[edge] = count
                count += 1

    def get_routing(self, action: Action) -> Routing:
        """
        Converts a destination routing to full routing
        Args:
            action: flat 1x(|V|*(|V|-1)*out_edges)
        Returns:
            A fully specified routing (dims 0: flows, 1: edges)
        """
        #TODO:
        # 1. Read into list of list of np arrays
        # 2. softmax the arrays
        # 3. Insert into "full" routing

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
            for j, edge in enumerate(self.graph.edges()):
                full_routing[i][j] = \
                    softmaxed_routing[edge[0]][dst][self.edge_index[edge]]

        return full_routing


class DDREnvSoftmin(DDREnv):
    """
    DDR Env where softmin routing is used (from Learning to Route with
    Deep RL paper). Routing is a single weight per edge, transformed to
    splitting ratios for input to the optimizer calculation.
    """

    def __init__(self,
                 dm_generator_getter: Callable[
                     [],
                     Generator[Demand, None, None]],
                 dm_memory_length: int,
                 graph: nx.DiGraph,
                 gamma: float = 2):
        super().__init__(dm_generator_getter, dm_memory_length, graph)

        # Precompute list of flows for use in routing translation
        num_nodes = self.graph.number_of_nodes()
        self.flows = [(i, j) for i in range(num_nodes) for j in range(num_nodes)
                      if i != j]
        self.gamma = gamma

        self.action_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(graph.number_of_edges(),))

        # Indices of the edges for lookup in routing translation
        # Takes each edge of graph to an index under its source node
        self.edge_index = {}
        for node in range(graph.number_of_nodes()):
            count = 0
            for edge in graph.out_edges(node):
                self.edge_index[edge] = count
                count += 1

    def get_routing(self, action: Action) -> Routing:
        """
        Converts a softmin routing to full routing
        Args:
            action: dims 0: edges
        Returns:
            A fully specified routing (dims 0: flows, 1: edges)
        """
        num_edges = self.graph.number_of_edges()
        full_routing = np.zeros((len(self.flows), num_edges), dtype=np.float32)
        softmin_edge_weights = np.zeros(num_edges)

        for i in range(self.graph.number_of_nodes()):
            out_edge_ids = [self.edge_index[e] for e in self.graph.out_edges(i)]
            out_edge_weights = [action[i] for i in out_edge_ids]
            softmin_weights = self.softmin(out_edge_weights)
            for j, id in enumerate(out_edge_ids):
                softmin_edge_weights[id] = softmin_weights[j]

        for i, flow in enumerate(self.flows):
            for j, edge in enumerate(self.graph.edges()):
                full_routing[i][j] = softmin_edge_weights[j]  # Surely wrong?

        return full_routing

    def softmin(self, array: List[float]) -> List[float]:
        """
        Calculates and returns the softmin of an np array
        Args:
            array: a list of floats
        Returns:
            a list of floats of the same size
        """
        exponentiated = [np.exp(-self.gamma * i) for i in array]
        total = sum(exponentiated)
        return [np.exp(-self.gamma * i) / total for i in exponentiated]
