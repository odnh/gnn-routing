from typing import Tuple, List, Dict, Callable, Generator, Type
import gym
import networkx as nx
import numpy as np

import demand_matrices
import max_link_utilisation

Routing = Type[np.ndarray]
Demand = Type[np.ndarray]
DMMemory = List[np.ndarray]
Action = Type[np.ndarray]

class DDREnv(gym.Env):
    """
    Gym env for data driven routing

    Observations are: last k routing and DMs
    Actions are: a routing (standard version: dest splitting ratios, softmin
    version: edge weigths)

    Actions are fully specified (destination based) routing. Subclass to
    either take a less specified version and transform otherwise learner will
    need to change too
    Rewards are: utilisation under routing compared to maximum utilisation
    """

    def __init__(self,
                 dm_generator_getter: Callable[
                     [],
                     Generator[Demand, None, None]],
                 dm_memory_length: int,
                 graph: nx.Graph):
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
        self.dm_memory: List[Demand] = []
        self.graph = graph
        self.done = False

    def step(self, action) -> Tuple[DMMemory, float, bool, Dict[None, None]]:
        """
        Args:
          action: a routing this is a fully specified routing
        Returns:
          history of dms and the other bits and pieces expected (use np.stack
          on the history for training)
        """
        # Check if sequence is exhausted
        if self.done:
            return (self.dm_memory.copy(), 0.0, self.done, dict())

        # update dm and history
        new_dm = next(self.dm_generator, None)
        if new_dm == None:
            self.done = True
            return (self.dm_memory.copy(), 0.0, self.done, dict())
        else:
            self.dm_memory.append(new_dm)
            if len(self.dm_memory) > self.dm_memory_length:
                self.dm_memory.pop(0)
            routing = self.get_routing(action)
            reward = self.get_reward(routing)
        return (self.dm_memory.copy(), reward, self.done, dict())

    def reset(self) -> DMMemory:
        self.dm_generator = self.dm_generator_getter()
        self.dm_memory = [next(self.dm_generator)]
        self.done = False
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


class DDREnvSoftmin(DDREnv):
    """
    DDR Env where all softmin routing is used (from Learning to Route with
    Deep RL paper). Routing is a single weight per edge, transformed to
    splitting ratios for input to the optimizer calculation.
    """
    def get_routing(self, edge_weights) -> Routing:
        pass
