# Sections that need implementing:
# 1. Actual environment providing obs and rewards for actions
# 2. Different deman matrix generation strategies
# 3. A simple way to couple the above two parts

import gym
from gym import error, spaces, utils
from gym.utils import seeding

class DDREnv(gym.Env):
    """
    Gym env for data driven routing
    Observations are: last k routing and DMs
    Actions are: a routing (standard version: dest splitting ratios, softmin version: edge weigths)
    Rewards are: utilisation compared to maximum utilisation
    """

    def __init__(self, dm_generator: Callable[[],Generator[np.ndarray]], dm_memory_length: int, graph: nerworkx.Graph):
        self.dm_generator_getter = dm_generator_getter
        self.dm_generator = dm_generator_getter()
        self.dm_memory_length = dm_memory_length
        self.dm_memory = []
        self.graph = graph

    def step(self, action: routing) -> Tuple[List[np.ndarray], float, bool, Dict[]]:
        # update dm and history
        new_dm = next(dm_generator)
        self.dm_memory.append(new_dm)
        if (len(self.dm_memory_length) > self.dm_memory_length):
            self.dm_memory.pop(0)
        routing = get_routing(action)
        reward = get_reward(routing)
        # work out when to set done
        return (self.dm_memory.copy(), reward, False, dict())

    def reset(self) -> List[np.ndarray]:
        self.dm_generator = self.dm_generator_getter()
        self.dm_memory = [next(self.dm_generator)]
        return self.dm_memory.copy()

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def get_reward(routing) -> float:
        utilisation = calc_utilisation(self.graph, self.dm_memory[0], routing)
        opt_utilisation = calc_opt_utilisation(self.graph, self.dm_memory[0])
        return -(utilisation/opt_utilisation)

    def calc_opt_utilisation(graph, dm):
        pass

class DDREnvDestSplitting(DDREnv):
    def get_routing(splitting_ratios) -> routing:
        pass

class DDREnvSoftmin(DDREnv):
    def get_routing(edge_weights) -> routing:
        pass

# Need to add
#  1. Sample dm generator
#  2. Function to calc OPT and u
