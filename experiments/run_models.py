import argparse
import datetime
import multiprocessing
import os
from typing import List, Dict, Tuple

import networkx as nx
import yaml
from ddr_learning_helpers import graphs

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import gym
import gym_ddr.envs.demand_matrices as dm
from gym_ddr.envs.max_link_utilisation import MaxLinkUtilisation
import numpy as np
from stable_baselines import PPO2
from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines_ddr.gnn_policy import MlpDdrPolicy, MlpLstmDdrPolicy, \
    GnnDdrPolicy, GnnDdrIterativePolicy, GnnLstmDdrPolicy, \
    GnnLstmDdrIterativePolicy


def run_model(env_name: str, policy: ActorCriticPolicy, graph: nx.DiGraph,
              demands: List[List[Tuple[np.ndarray, float]]], model_path: str,
              replay_steps: int = 10,
              env_kwargs: Dict = {}, policy_kwargs: Dict = {},
              hyperparameters: Dict = {},
              parallelism: int = 4):
    oblivious_routing = None  # yates.get_oblivious_routing(graph)

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
    if env_name == 'ddr-iterative-v0':
        reward_inc = 0
        for i in range(replay_steps):
            action, state = model.predict(obs, state=state, deterministic=True)
            obs, reward, done, info = env.step(action)
            print(reward)
            print(info)
            if sum(info['edge_set']) == 0:
                reward_inc += 1
                total_rewards += info['real_reward']
        print("Mean reward: ", total_rewards / reward_inc)
    else:
        for i in range(replay_steps):
            action, state = model.predict(obs, state=state, deterministic=True)
            obs, reward, done, info = env.step(action)
            print(reward)
            print(info)
            total_rewards += reward
        print("Mean reward: ", total_rewards / replay_steps)
    env.close()


def demands_from_args(args: Dict, graph: nx.DiGraph) -> List[
    List[Tuple[np.ndarray, float]]]:
    num_demands = graph.number_of_nodes() * (graph.number_of_nodes() - 1)
    dm_generator_getter = lambda seed: dm.cyclical_sequence(
        # A function that returns a generator for a sequence of demands
        lambda rs_l: dm.bimodal_demand(num_demands, rs_l),
        args['sequence_length'], args['cycle_length'], args['sparsity'],
        seed=seed)
    mlu = MaxLinkUtilisation(graph)
    demand_sequences = map(dm_generator_getter, args['demand_seeds'])
    demands_with_opt = [[(demand, mlu.opt(demand)) for demand in sequence] for
                        sequence in demand_sequences]
    return demands_with_opt


def policy_from_args(args: Dict, graph: nx.DiGraph) -> Tuple[
    ActorCriticPolicy, Dict]:
    policy_kwargs = {}
    dm_memory_length = 1 if args['lstm'] else args['memory_length']
    if args['policy'] == 'gnn':
        policy = GnnLstmDdrPolicy if args['lstm'] else GnnDdrPolicy
        policy_kwargs = {'network_graph': graph,
                         'dm_memory_length': dm_memory_length,
                         'vf_arch': args['vf_arch'],
                         }
    elif args['policy'] == 'iter':
        policy = GnnLstmDdrIterativePolicy if args[
            'lstm'] else GnnDdrIterativePolicy
        policy_kwargs = {'network_graph': graph,
                         'dm_memory_length': dm_memory_length,
                         'vf_arch': args['vf_arch']}
    else:
        policy = MlpLstmDdrPolicy if args['lstm'] else MlpDdrPolicy

    return policy, policy_kwargs


def graph_from_args(args: Dict) -> nx.DiGraph:
    if args['graph']:
        graph = graphs.topologyzoo(args['graph'], 10000)
    else:
        graph = graphs.basic()
    return graph


def env_kwargs_from_args(args: Dict) -> Dict:
    env_kwargs = {}
    if args['memory_length']:
        env_kwargs['dm_memory_length'] = args['memory_length']
    return env_kwargs


def argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DDR experiment")
    parser.add_argument('-c', action='store', dest='config',
                        help="Config file to read. Overrides all other options")
    parser.add_argument('-hy', action='store', dest='hyperparameters',
                        default=None,
                        help="Hyperprameter config file to read.")
    parser.add_argument('-e', action='store', dest='env',
                        default='ddr-iterative-v0',
                        help="Name of environment to train on")
    parser.add_argument('-p', action='store', dest='policy',
                        default='iter',
                        help="Name of the policy to use")
    parser.add_argument('-g', action='store', dest='graph',
                        help="Name of graph to train on")
    parser.add_argument('-t', action='store', dest='timesteps', type=int,
                        default=10000,
                        help="Number of timesteps of training to perform")
    parser.add_argument('-m', action='store', dest='memory_length', type=int,
                        default=10,
                        help="Demand matrix memory length")
    parser.add_argument('-s', nargs='+', dest='demand_seeds', type=int,
                        default=[1],
                        help="Seeds for demand sequences")
    parser.add_argument('-q', action='store', dest='cycle_length', type=int,
                        default=5,
                        help="Length of cycles in demand matrix sequence")
    parser.add_argument('-sp', action='store', dest='sparsity', type=float,
                        default=0.0,
                        help="Demand matrix sparsity")
    parser.add_argument('-l', action='store', dest='sequence_length', type=int,
                        default=50,
                        help="Demand matrix sequence length")
    parser.add_argument('-v', action='store', dest='vf_arch', default='graph',
                        help="Value function architecture")
    parser.add_argument('-sd', action='store', dest='seed', type=int,
                        default=int(datetime.datetime.now().timestamp()),
                        help="Random seed for the run")
    parser.add_argument('-pl', action='store', dest='parallelism', type=int,
                        default=4,
                        help="Number of envs to run in parallel")
    parser.add_argument('-mn', action='store', dest='model_name', type=int,
                        default=None, help="Name to save model as")
    parser.add_argument('-ln', action='store', dest='log_name', default=None,
                        help="Name for tensorboboard log")
    parser.add_argument('-lstm', action='store_true', dest='lstm',
                        help="Whether to use an lstm layer (default is false)")
    parser.add_argument('-rs', action='store', dest='Replay steps',
                        help="Number of steps to take when replaying the env")
    parser.add_argument('-mp', action='store', dest='model_path',
                        help="Path to the stored model to be loaded.")
    return parser


def args_from_config(args: Dict) -> Dict:
    with open(args['config'], 'r') as stream:
        config = yaml.safe_load(stream)
    return config


def read_hyperparameters(args: Dict) -> Dict:
    hyperparameters = {}
    if args['hyperparameters']:
        with open(args['hyperparameters'], 'r') as stream:
            hyperparameters = yaml.safe_load(stream)
    return hyperparameters


def seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


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
