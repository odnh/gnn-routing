import tensorflow as tf
from gym_ddr.envs.max_link_utilisation import MaxLinkUtilisation

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import gym
import gym_ddr.envs.demand_matrices as dm
import numpy as np
from ddr_learning_helpers import graphs, yates
from stable_baselines.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy

if __name__ == "__main__":
    # load/generate graph
    graph = graphs.topologyzoo("TLex", 10000)
    # graph = graphs.basic()

    ## ENV PARAMETERS
    rs = np.random.RandomState()  # Random state
    dm_memory_length = 10  # Length of memory of dms in each observation
    num_demands = graph.number_of_nodes() * (graph.number_of_nodes() - 1)  # Demand matrix size (dependent on graph size
    dm_generator_getter = lambda seed: dm.cyclical_sequence(  # A function that returns a generator for a sequence of demands
        lambda rs_l: dm.bimodal_demand(num_demands, rs_l), 20, 5, 0.0, seed=seed)
    mlu = MaxLinkUtilisation(graph)  # Friendly max link utilisation class
    demand_sequences = map(dm_generator_getter, [28])
    demands_with_opt = [[(demand, mlu.opt(demand)) for demand in sequence] for  # Merge opt calculations into the demand sequence
                        sequence in demand_sequences]

    oblivious_routing = None  # yates.get_oblivious_routing(graph)

    # make env
    env = lambda: gym.make('ddr-softmin-v0',
                           dm_sequence=demands_with_opt,
                           dm_memory_length=dm_memory_length,
                           graph=graph,
                           oblivious_routing=oblivious_routing)

    vec_env = SubprocVecEnv([env, env, env, env], start_method="spawn")
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
