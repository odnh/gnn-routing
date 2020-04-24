import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import gym
import gym_ddr.envs.demand_matrices as dm
import numpy as np
from ddr_learning_helpers import graphs, yates
from stable_baselines.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy

# load/generate graph
graph = graphs.topologyzoo("TLex", 10000)

# set env parameters
rs = np.random.RandomState()
dm_memory_length = 10
num_demands = graph.number_of_nodes() * (graph.number_of_nodes() - 1)
num_edges = graph.number_of_edges()
dm_generator_getter = lambda: dm.cyclical_sequence(
    lambda rs_l: dm.bimodal_demand(num_demands, rs_l), 50, 5, 0.0, seed=32)
# dm_generator_getter = lambda: dm.average_sequence(
#     lambda: dm.gravity_demand(graph), 40, 5, 0.4, rs)

oblivious_routing = None#yates.get_oblivious_routing(graph)

# make env
env = lambda: gym.make('ddr-softmin-v0',
                       dm_sequence=[list(dm_generator_getter())],
                       dm_memory_length=dm_memory_length,
                       graph=graph,
                       oblivious_routing=oblivious_routing)

vec_env = SubprocVecEnv([env, env, env, env])
# Try with and without. May interfere with iter
normalised_env = VecNormalize(vec_env, training=True, norm_obs=True,
                              norm_reward=False)

# make model
model = PPO2(MlpPolicy,
             normalised_env,
             verbose=1,
             tensorboard_log="./gnn_tensorboard/")

# learn
model.learn(total_timesteps=100000, tb_log_name="mlp_softmin_basic")
model.save("./model_mlp_softmin_basic")

# use
obs = normalised_env.reset()
for i in range(61):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(rewards)
    print(info)
