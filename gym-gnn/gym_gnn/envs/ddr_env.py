from typing import Tuple, List, Dict, Callable, Generator, Type
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import networkx as nx
import numpy as np

from demand_matrices import random_demand, gravity_demand, bimodal_demand
import max_link_utilisation

# Definition of the Routing type (will probablly change)
Routing = Type[np.ndarray]
DemandMatrix = Type[np.ndarray]
DMMemory = List[np.ndarray]
Action = Type[np.ndarray]

class DDREnv(gym.Env):
    """
    Gym env for data driven routing
    Observations are: last k routing and DMs
    Actions are: a routing (standard version: dest splitting ratios, softmin
    version: edge weigths)
    Actions are fully specified routing. Subclass to either take a less
    specified version and transform otherwise learner will need to change too
    Rewards are: utilisation under routing compared to maximum utilisation
    """

    def __init__(self, dm_generator_getter:
            Callable[[],Generator[DemandMatrix, None, None]],
            dm_memory_length: int, graph: nx.Graph):
        """
        Args:
          dm_generator_getter: a function that returns a genrator for demand
                               matrices (so can reset)
          dm_memory_length: the length of the dm history we should train on
          graph: the graph we will be routing over
        """
        self.dm_generator_getter = dm_generator_getter
        self.dm_generator = dm_generator_getter()
        self.dm_memory_length = dm_memory_length
        self.dm_memory: List[DemandMatrix] = []
        self.graph = graph

    def step(self, action) -> Tuple[DMMemory, float, bool, Dict[None, None]]:
        """
        Args:
          action: a routing this is a fully specified routing
        Returns:
          history of dms and the other bits and pieces expected (use np.stack
          on the history for training)
        """
        # update dm and history
        new_dm = next(self.dm_generator)
        self.dm_memory.append(new_dm)
        if (len(self.dm_memory) > self.dm_memory_length):
            self.dm_memory.pop(0)
        routing = self.get_routing(action)
        reward = self.get_reward(routing)
        # work out when to set done
        return (self.dm_memory.copy(), reward, False, dict())

    def reset(self) -> DMMemory:
        self.dm_generator = self.dm_generator_getter()
        self.dm_memory = [next(self.dm_generator)]
        return self.dm_memory.copy()

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def get_routing(self, action: Action) -> Routing:
        """
        Subclass to use different actions, assumes action is a routing in base
        case
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
        return -(utilisation/opt_utilisation)


class DDREnvDestSplitting(DDREnv):
    """
    DDR Env where all routes are destination based (i.e. each edge has
    ratios for traffic based only on destination)
    """
    def get_routing(self, splitting_ratios) -> Routing:
        pass

class DDREnvSoftmin(DDREnv):
    """
    DDR Env where all softmin routing is used (from Leanring to Route with
    Deep RL paper). Routing is a songle weight per edge, transformed to
    splitting ratios for input to the optimizer calculation.
    """
    def get_routing(self, edge_weights) -> Routing:
        pass
