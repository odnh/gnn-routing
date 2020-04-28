import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import gym
import gym_ddr.envs.demand_matrices as dm
from gym_ddr.envs.max_link_utilisation import MaxLinkUtilisation
import numpy as np
from ddr_learning_helpers import graphs
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import PPO2
from stable_baselines_ddr.gnn_policy import GnnDdrPolicy

from ray import tune


def tune_ddr_gnn(config, reporter):
    # load/generate graph
    # graph = graphs.topologyzoo("TLex", 10000)
    graph = graphs.basic()

    ## ENV PARAMETERS
    rs = np.random.RandomState()  # Random state
    dm_memory_length = 10  # Length of memory of dms in each observation
    steps_in_episode = 40  # how many steps in one episode
    num_demands = graph.number_of_nodes() * (
                graph.number_of_nodes() - 1)  # Demand matrix size (dependent on graph size
    dm_generator_getter = lambda seed: dm.cyclical_sequence(
        # A function that returns a generator for a sequence of demands
        lambda rs_l: dm.bimodal_demand(num_demands, rs_l),
        steps_in_episode + dm_memory_length, 5, 0.0, seed=seed)
    # demand_sequences = [list(dm_generator_getter()) for i in range(2)]  # Collect the generator into a sequence
    mlu = MaxLinkUtilisation(graph)  # Friendly max link utilisation class
    demand_sequences = map(dm_generator_getter, [32, 32])
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
    # Try with and without. May interfere with iter
    normalised_env = VecNormalize(vec_env, training=True, norm_obs=True,
                                  norm_reward=False)

    # make model
    model = PPO2(GnnDdrPolicy,
                 normalised_env,
                 verbose=1,
                 policy_kwargs={'network_graph': graph,
                                'dm_memory_length': dm_memory_length,
                                'vf_arch': "graph"},
                 **config)

    # learn
    model.learn(total_timesteps=10000)

    total_rewards = 0
    obs = normalised_env.reset()
    for i in range(steps_in_episode):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = normalised_env.step(action)
        total_rewards += sum(rewards)
    reporter(mean_reward=total_rewards / steps_in_episode)
    normalised_env.close()


if __name__ == "__main__":
    analysis = tune.run(
        tune_ddr_gnn,
        config={
            "batch_size": tune.grid_search([32, 64, 128, 256]),
            "n_steps": tune.grid_search(
                [16, 32, 64, 128, 256, 512, 1024, 2048]),
            "gamma": tune.grid_search(
                [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]),
            "learning_rate": tune.grid_search([1e-5, 1]),
            "ent_coef": tune.grid_search([0.00000001, 0.1]),
            "cliprange": tune.grid_search([0.1, 0.2, 0.3, 0.4]),
            "noptepochs": tune.grid_search([1, 5, 10, 20, 30, 50]),
            "lam": tune.grid_search([0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0]),
        },
        resources_per_trial={'cpu': 1, 'gpu': 1})

    print("Best config: ", analysis.get_best_config(metric="mean_reward"))
