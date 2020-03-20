from typing import Tuple, List, Dict, Callable, Generator, Type
import gym
import networkx as nx
import numpy as np

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

    NB: actions and observations are flattened for external view but retain
    their shape internally and when used in the optimisation step
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
            low=0.0,  # TODO: should be 0 but gaussian model requires otherwise
            high=1.0,
            shape=(graph.number_of_nodes() * (
            graph.number_of_nodes() - 1) * graph.number_of_edges(),))
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(dm_memory_length *
            graph.number_of_nodes() * (graph.number_of_nodes() - 1),))

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
        routing in base case
        """
        return action

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
        return np.concatenate(self.dm_memory).ravel()


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
        self.action_space = gym.spaces.Box( #TODO: set this properly
            low=0.0,
            high=1.0,
            shape=(graph.number_of_nodes() * (
                    graph.number_of_nodes() - 1) * graph.number_of_edges(),))

    def get_routing(self, action: Action) -> Routing:
        pass


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
                 graph: nx.DiGraph):
        super().__init__(dm_generator_getter, dm_memory_length, graph)
        self.action_space = gym.spaces.Box( #TODO: set this properly
            low=0.0,
            high=1.0,
            shape=(graph.number_of_nodes() * (
                    graph.number_of_nodes() - 1) * graph.number_of_edges(),))

    def get_routing(self, action: Action) -> Routing:
        pass
