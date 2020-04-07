import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import gym
import numpy as np
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy

from ddr_learning_helpers import graphs
import gym_ddr.envs.demand_matrices as dm

# load/generate graph
graph = graphs.topologyzoo("TLex", 10000)

# set env parameters
rs = np.random.RandomState()
dm_memory_length = 10
num_demands = graph.number_of_nodes() * (graph.number_of_nodes() - 1)
num_edges = graph.number_of_edges()
dm_generator_getter = lambda: dm.cyclical_sequence(
    lambda: dm.bimodal_demand(num_demands, rs), 40, 5, 0.0, rs)

# make env
env = gym.make('ddr-softmin-v0', dm_generator_getter=dm_generator_getter,
               dm_memory_length=dm_memory_length, graph=graph)

# make model
model = PPO2(MlpPolicy, env, verbose=1)

# learn
model.learn(total_timesteps=10000)

# use
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(rewards)
