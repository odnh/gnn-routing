import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import gym
import gym_ddr.envs.demand_matrices as dm
from gym_ddr.envs.max_link_utilisation import MaxLinkUtilisation
import numpy as np
from ddr_learning_helpers import graphs
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines_ddr.gnn_policy import GnnDdrPolicy


def tuneable(config):
    # load/generate graph
    # graph = graphs.topologyzoo("TLex", 10000)
    graph = graphs.basic()

    ## ENV PARAMETERS
    rs = np.random.RandomState()  # Random state
    num_steps = 20
    dm_memory_length = 10  # Length of memory of dms in each observation
    num_demands = graph.number_of_nodes() * (
                graph.number_of_nodes() - 1)  # Demand matrix size (dependent on graph size
    dm_generator_getter = lambda seed: dm.cyclical_sequence(
        # A function that returns a generator for a sequence of demands
        lambda rs_l: dm.bimodal_demand(num_demands, rs_l),
        num_steps + dm_memory_length, 5, 0.0, seed=seed)
    mlu = MaxLinkUtilisation(graph)  # Friendly max link utilisation class
    demand_sequences = map(dm_generator_getter, [32])
    demands_with_opt = [[(demand, mlu.opt(demand)) for demand in sequence] for
                        # Merge opt calculations into the demand sequence
                        sequence in demand_sequences]

    oblivious_routing = None  # yates.get_oblivious_routing(graph)

    # make env
    env = lambda: gym.make('ddr-softmin-v0',
                           dm_sequence=demands_with_opt,
                           dm_memory_length=dm_memory_length,
                           graph=graph,
                           oblivious_routing=oblivious_routing)

    vec_env = DummyVecEnv([env])

    # sort out batch parameter
    config['nminibatches'] = max(int(config['n_steps'] / config['batch_size']),
                                 1)
    del config['batch_size']

    # make model
    model = PPO2(GnnDdrPolicy,
                 vec_env,
                 verbose=0,
                 policy_kwargs={'network_graph': graph,
                                'dm_memory_length': dm_memory_length,
                                'vf_arch': "graph"},
                 **config)

    # learn
    model.learn(total_timesteps=10000, tb_log_name="gnn_softmin_basic")
    model.save("./model_gnn_softmin_basic")

    # use
    obs = vec_env.reset()
    total_reward = 0.0
    for i in range(num_steps * graph.number_of_edges()):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        total_reward += rewards[0]

    vec_env.close()

    return total_reward
