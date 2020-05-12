import multiprocessing
import os
from typing import List, Dict, Tuple

import networkx as nx
from ddr_learning_helpers import routing_baselines

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import gym
import numpy as np
from stable_baselines import PPO2
from stable_baselines.common.policies import ActorCriticPolicy

from run_experiments import *

def run_model(env_name: str, policy: ActorCriticPolicy, graph: nx.DiGraph,
              demands: List[List[Tuple[np.ndarray, float]]], model_path: str,
              replay_steps: int = 10,
              env_kwargs: Dict = {}, policy_kwargs: Dict = {},
              hyperparameters: Dict = {},
              parallelism: int = 4):
    # oblivious_routing = yates.get_oblivious_routing(graph)
    oblivious_routing = routing_baselines.shortest_path_routing(graph)
    print(oblivious_routing)

    # make env
    env = gym.make(env_name,
                   dm_sequence=demands,
                   graph=graph,
                   oblivious_routing=oblivious_routing,
                   **env_kwargs)

    # load
    model = PPO2.load(model_path)

    # execute
    obs = env.reset()
    state = None
    total_rewards = 0
    oblivious_rewards = 0
    if env_name == 'ddr-iterative-v0':
        reward_inc = 0
        for i in range(replay_steps-1):
            action, state = model.predict(obs, state=state, deterministic=True)
            obs, reward, done, info = env.step(action)
            print(reward)
            print(info)
            if sum(info['edge_set']) == 0:
                reward_inc += 1
                total_rewards += info['real_reward']
                oblivious_rewards += info['oblivious_utilisation'] / info['opt_utilisation']
        print("Mean reward: ", total_rewards / reward_inc)
        print("Mean oblivious reward: ", oblivious_rewards / reward_inc)
    else:
        for i in range(replay_steps-1):
            action, state = model.predict(obs, state=state, deterministic=True)
            obs, reward, done, info = env.step(action)
            print(reward)
            print(action)
            print(info)
            total_rewards += reward
            oblivious_rewards += info['oblivious_utilisation'] / info['opt_utilisation']
        print("Mean reward: ", total_rewards / (replay_steps-1))
        print("Mean oblivious reward: ", oblivious_rewards / (replay_steps-1))
    env.close()


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    parser = argparser()
    args = vars(parser.parse_args())

    if args['config']:
        args = args_from_config(args)

    args['model_name'] = args['model_name'] if args[
        'model_name'] else "{}_{}_{}".format(args['env'], args['policy'],
                                             args['graph'])
    args['log_name'] = args['log_name'] if args['log_name'] else args[
        'model_name']

    print("Run configuration:")
    print(args)
    seed(args['seed'])

    hyperparameters = read_hyperparameters(args)
    graph = graph_from_args(args)
    policy, policy_kwargs = policy_from_args(args, graph)
    demands = demands_from_args(args, graph)
    env_kwargs = env_kwargs_from_args(args)

    run_model(args['env'], policy, graph, demands, args['model_path'],
              args['replay_steps'], env_kwargs, policy_kwargs, hyperparameters,
              args['parallelism'])
