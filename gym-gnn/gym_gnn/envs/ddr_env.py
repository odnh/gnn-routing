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
    Observations are:
    Actions are:
    Rewards are:
    """

    def __init__(self):
        pass
    def step(self, action):
        pass
    def reset(self):
        pass
    def render(self, mode='human'):
        pass
    def close(self):
        pass
