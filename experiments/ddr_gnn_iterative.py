import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import gym
import gym_ddr.envs.demand_matrices as dm
import numpy as np
from ddr_learning_helpers import graphs
from stable_baselines import PPO2
from stable_baselines_ddr.gnn_policy import GnnDdrIterativePolicy

# load/generate graph
#graph = graphs.topologyzoo("TLex", 10000)
graph = graphs.basic()

# set env parameters
rs = np.random.RandomState()
dm_memory_length = 10
num_demands = graph.number_of_nodes() * (graph.number_of_nodes() - 1)
nn = graph.number_of_nodes()
lend = len([(i, j) for i in range(nn) for j in range(nn) if i != j])
num_edges = graph.number_of_edges()
# dm_generator_getter = lambda: dm.cyclical_sequence(
#     lambda: dm.bimodal_demand(num_demands, rs), 40, 5, 0.0, rs)
dm_generator_getter = lambda: dm.average_sequence(
    lambda: dm.gravity_demand(graph), 40, 5, 0.4, rs)

# make env
env = gym.make('ddr-iterative-v0', dm_generator_getter=dm_generator_getter,
               dm_memory_length=dm_memory_length, graph=graph)

# make model
model = PPO2(GnnDdrIterativePolicy, env, verbose=1,
             policy_kwargs={'network_graph': graph,
                            'dm_memory_length': dm_memory_length},
             tensorboard_log="./gnn_tensorboard/")

# learn
model.learn(total_timesteps=100000)

# use
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(rewards)
    env.render()
