import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import gym
import gym_ddr.envs.demand_matrices as dm
import numpy as np
from ddr_learning_helpers import graphs
from stable_baselines import PPO2
from stable_baselines_ddr.gnn_policy import GnnDdrPolicy

# load/generate graph
graph = graphs.topologyzoo("TLex", 10000)
#graph = graphs.basic()

# set env parameters
rs = np.random.RandomState()
dm_memory_length = 5
num_demands = graph.number_of_nodes() * (graph.number_of_nodes() - 1)
num_edges = graph.number_of_edges()
dm_generator_getter = lambda: dm.cyclical_sequence(
    lambda: dm.bimodal_demand(num_demands, rs), 40, 15, 0.3, rs)

# make env
env = gym.make('ddr-softmin-v0', dm_generator_getter=dm_generator_getter,
               dm_memory_length=dm_memory_length, graph=graph)

# make model
model = PPO2(GnnDdrPolicy, env, verbose=1,
             policy_kwargs={'network_graph': graph}, tensorboard_log="./gnn_tensorboard/")

# learn
model.learn(total_timesteps=100000)

# use
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
