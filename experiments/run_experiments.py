import argparse
import datetime
import json
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
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.policies import ActorCriticPolicy, LstmPolicy
from stable_baselines_ddr.policies import MlpDdrPolicy, GnnDdrPolicy, \
    GnnDdrIterativePolicy


def true_reward_callback(locals_, globals_):
    self_ = locals_['self']
    if len(self_.ep_info_buf) != 0:
        summary = tf.Summary(
            value=[tf.Summary.Value(tag='real_epmrw',
                                    simple_value=self_.ep_info_buf[-1]['r'])])
        locals_['writer'].add_summary(summary, self_.num_timesteps)
    return True


def run_experiment(env_name: str, policy: ActorCriticPolicy, graph: nx.DiGraph,
                   demands: List[List[Tuple[np.ndarray, float]]],
                   env_kwargs: Dict = {}, policy_kwargs: Dict = {},
                   hyperparameters: Dict = {}, timesteps: int = 10000,
                   parallelism: int = 4, model_name: str = "",
                   log_name: str = "", replay_steps: int = 10):
    oblivious_routing = None  # yates.get_oblivious_routing(graph)

    # make env
    env = lambda: gym.make(env_name,
                           dm_sequence=demands,
                           graph=graph,
                           oblivious_routing=oblivious_routing,
                           **env_kwargs)
    vec_env = SubprocVecEnv([env for _ in range(parallelism)],
                            start_method="spawn")

    # make model
    model = PPO2(policy,
                 vec_env,
                 cliprange_vf=-1,
                 verbose=1,
                 policy_kwargs=policy_kwargs,
                 tensorboard_log="./gnn_tensorboard/",
                 **hyperparameters)

    # learn
    if env_name == 'ddr-iterative-v0':
        model.learn(total_timesteps=timesteps, tb_log_name=log_name,
                    callback=true_reward_callback)
    else:
        model.learn(total_timesteps=timesteps, tb_log_name=log_name)

    # save it
    model.save(model_name)

    # try it out
    obs = vec_env.reset()
    state = None
    total_rewards = 0
    if env_name == 'ddr-iterative-v0':
        reward_inc = 0
        for i in range(replay_steps - 1):
            action, state = model.predict(obs, state=state, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            print(reward)
            print(info)
            if sum(info[0]['edge_set']) == 0:
                reward_inc += 1
                total_rewards += info[0]['real_reward']
        print("Mean reward: ", total_rewards / reward_inc)
    else:
        for i in range(replay_steps - 1):
            action, state = model.predict(obs, state=state, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            print(reward)
            print(info)
            total_rewards += reward[0]
        print("Mean reward: ", total_rewards / (replay_steps - 1))
    vec_env.close()


def demands_from_args(args: Dict, graph: nx.DiGraph) -> List[
    List[Tuple[np.ndarray, float]]]:
    num_demands = graph.number_of_nodes() * (graph.number_of_nodes() - 1)
    # because the first n demands are used to build the history
    sequence_length = args['sequence_length'] + args['memory_length']
    dm_generator_getter = lambda seed: dm.cyclical_sequence(
        # A function that returns a generator for a sequence of demands
        lambda rs_l: dm.bimodal_demand(num_demands, rs_l),
        sequence_length, args['cycle_length'], args['sparsity'],
        seed=seed)
    mlu = MaxLinkUtilisation(graph)
    demand_sequences = map(dm_generator_getter, args['demand_seeds'])
    demands_with_opt = [[(demand, mlu.opt(demand)) for demand in sequence] for
                        sequence in demand_sequences]
    return demands_with_opt


def policy_from_args(args: Dict, graph: nx.DiGraph) -> Tuple[
    ActorCriticPolicy, Dict]:
    dm_memory_length = args['memory_length']
    iterations = args['gnn_iterations'] if 'gnn_iterations' in args else 10
    if args['policy'] == 'gnn':
        policy = GnnDdrPolicy
        policy_kwargs = {'network_graph': graph,
                         'dm_memory_length': dm_memory_length,
                         'vf_arch': args['vf_arch'],
                         'iterations': iterations
                         }
    elif args['policy'] == 'iter':
        policy = GnnDdrIterativePolicy
        policy_kwargs = {'network_graph': graph,
                         'dm_memory_length': dm_memory_length,
                         'vf_arch': args['vf_arch'],
                         'iterations': iterations
                         }
    elif args['policy'] == 'lstm':
        policy = LstmPolicy
        policy_kwargs = {'feature_extraction': 'mlp'}
    else:
        policy = MlpDdrPolicy
        policy_kwargs = {'network_graph': graph}

    return policy, policy_kwargs


def graph_from_args(args: Dict) -> nx.DiGraph:
    if args['graph']:
        graph = graphs.topologyzoo(args['graph'], 10000)
    else:
        graph = graphs.basic()
    return graph


def env_kwargs_from_args(args: Dict) -> Dict:
    env_kwargs = {}
    if 'memory_length' in args:
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
    parser.add_argument('-rs', action='store', dest='replay_steps',
                        help="Number of steps to take when replaying the env")
    parser.add_argument('-mp', action='store', dest='model_path',
                        help="Path to the stored model to be loaded.")
    parser.add_argument('-gi', action='store', dest='gnn_iterations',
                        help="Number of message passing iterations in gnn")
    return parser


def args_from_config(args: Dict) -> Dict:
    with open(args['config'], 'r') as stream:
        config = yaml.safe_load(stream)
    return config


def read_hyperparameters(args: Dict) -> Dict:
    hyperparameters = {}
    if args['hyperparameters']:
        with open(args['hyperparameters'], 'r') as stream:
            hyperparameters = json.load(stream)
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

    run_experiment(args['env'], policy, graph, demands, env_kwargs,
                   policy_kwargs, hyperparameters, args['timesteps'],
                   args['parallelism'], args['log_name'], args['model_name'],
                   args['replay_steps'])
