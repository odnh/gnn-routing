"""
A selection of functions for help in running experiments, tuning, models, and
training rom the command line.
"""
import argparse
import json
import multiprocessing
import os
from typing import List, Dict, Tuple

import gym
from ddr_learning_helpers import routing_baselines
from jsonlines import jsonlines
from stable_baselines import PPO2
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np
import networkx as nx
import yaml
from stable_baselines.common.policies import ActorCriticPolicy, LstmPolicy

from stable_baselines_ddr.policies import MlpDdrPolicy, GnnDdrPolicy, \
    GnnDdrIterativePolicy
from ddr_learning_helpers.graphs import from_graphspec
import gym_ddr.envs.demand_matrices as dm
from gym_ddr.envs.max_link_utilisation import MaxLinkUtilisation


def true_reward_callback(locals_, globals_):
    """
    Callback to report the real reward on Tensorboard for iterative training
    """
    self_ = locals_['self']
    if len(self_.ep_info_buf) != 0:
        summary = tf.Summary(
            value=[tf.Summary.Value(tag='real_epmrw',
                                    simple_value=self_.ep_info_buf[-1]['r'])])
        locals_['writer'].add_summary(summary, self_.num_timesteps)
    return True


def demands_from_args(args: Dict, graphs: List[nx.DiGraph]) -> List[List[
    List[Tuple[np.ndarray, float]]]]:
    """
    Uses program arguments to build demand sequences. Return is list of list of
    sequences paired with opt value, one set for each graph.
    """
    demands_per_graph = []
    for graph in graphs:
        num_demands = graph.number_of_nodes() * (graph.number_of_nodes() - 1)
        # because the first n demands are used to build the history
        sequence_length = args['sequence_length'] + args['memory_length']

        # Select sequence type
        if args['sequence_type'] == 'cyclical':
            dm_generator_getter = lambda seed: dm.cyclical_sequence(
                lambda rs_l: dm.bimodal_demand(num_demands, rs_l),
                sequence_length, args['cycle_length'], args['sparsity'],
                seed=seed)
        elif args['sequence_type'] == 'gravity':
            dm_generator_getter = lambda seed: dm.cyclical_sequence(
                lambda _: dm.gravity_demand(graph),
                sequence_length, args['cycle_length'], args['sparsity'],
                seed=seed)
        elif args['sequence_type'] == 'average':
            dm_generator_getter = lambda seed: dm.average_sequence(
                lambda rs_l: dm.bimodal_demand(num_demands, rs_l),
                sequence_length, args['cycle_length'], args['sparsity'],
                seed=seed)
        else:
            raise Exception("No such sequence type")

        mlu = MaxLinkUtilisation(graph)
        demand_sequences = map(dm_generator_getter, args['demand_seeds'])
        demands_with_opt = [[(demand, mlu.opt(demand)) for demand in sequence]
                            for
                            sequence in demand_sequences]
        demands_per_graph.append(demands_with_opt)
    return demands_per_graph


def policy_from_args(args: Dict, graphs: List[nx.DiGraph]) -> Tuple[
    ActorCriticPolicy, Dict]:
    """Uses program arguments to build policy network"""
    dm_memory_length = args['memory_length']
    iterations = args['gnn_iterations'] if 'gnn_iterations' in args else 10
    if args['policy'] == 'gnn':
        policy = GnnDdrPolicy
        policy_kwargs = {'network_graphs': graphs,
                         'dm_memory_length': dm_memory_length,
                         'vf_arch': args['vf_arch'],
                         'iterations': iterations
                         }
    elif args['policy'] == 'iter':
        policy = GnnDdrIterativePolicy
        policy_kwargs = {'network_graphs': graphs,
                         'dm_memory_length': dm_memory_length,
                         'vf_arch': args['vf_arch'],
                         'iterations': iterations
                         }
    elif args['policy'] == 'lstm':
        policy = LstmPolicy
        policy_kwargs = {'feature_extraction': 'mlp'}
    else:
        policy = MlpDdrPolicy
        policy_kwargs = {'network_graphs': graphs}

    return policy, policy_kwargs


def graphs_from_args(graphspecs: List[str]) -> List[nx.DiGraph]:
    """Uses program arguments to build graph"""
    graphs = []
    for graphspec in graphspecs:
        graphs.append(from_graphspec(graphspec))
    return graphs


def env_kwargs_from_args(args: Dict) -> Dict:
    """Uses program arguments to build env"""
    env_kwargs = {}
    if 'memory_length' in args:
        env_kwargs['dm_memory_length'] = args['memory_length']
    return env_kwargs


def argparser() -> argparse.ArgumentParser:
    """Builds argparser for program arguments"""
    parser = argparse.ArgumentParser(description="Run DDR experiment")
    parser.add_argument('-c', action='store', dest='config',
                        help="Path of config file to read. Other options"
                             "override these values.")
    parser.add_argument('-hy', action='store', dest='hyperparameters',
                        help="Hyperprameter (json) config file to read.")
    parser.add_argument('-e', action='store', dest='env',
                        help="Name of environment to train on")
    parser.add_argument('-p', action='store', dest='policy',
                        help="Name of the policy to use")
    parser.add_argument('-g', nargs='+', action='store', dest='graphs',
                        help="Name of graphs to train on")
    parser.add_argument('-t', action='store', dest='timesteps', type=int,
                        help="Number of timesteps of training to perform")
    parser.add_argument('-m', action='store', dest='memory_length', type=int,
                        help="Demand matrix memory length")
    parser.add_argument('-s', nargs='+', dest='demand_seeds', type=int,
                        help="Seeds for demand sequences")
    parser.add_argument('-q', action='store', dest='cycle_length', type=int,
                        help="Length of cycles in demand matrix sequence")
    parser.add_argument('-sp', action='store', dest='sparsity', type=float,
                        help="Demand matrix sparsity")
    parser.add_argument('-l', action='store', dest='sequence_length', type=int,
                        help="Demand matrix sequence length")
    parser.add_argument('-v', action='store', dest='vf_arch',
                        help="Value function architecture")
    parser.add_argument('-sd', action='store', dest='seed', type=int,
                        help="Random seed for the run")
    parser.add_argument('-pl', action='store', dest='parallelism', type=int,
                        help="Number of envs to run in parallel")
    parser.add_argument('-mn', action='store', dest='model_name', type=int,
                        help="Name to save model as")
    parser.add_argument('-ln', action='store', dest='log_name',
                        help="Name for tensorboard log")
    parser.add_argument('-rs', action='store', dest='replay_steps',
                        help="Number of steps to take when replaying the env")
    parser.add_argument('-mp', action='store', dest='model_path',
                        help="Path to the stored model to be loaded.")
    parser.add_argument('-gi', action='store', dest='gnn_iterations',
                        help="Number of message passing iterations in gnn")
    parser.add_argument('-o', action='store', dest='output_path',
                        help="Path of file to write output to")
    parser.add_argument('-tb', action='store', dest='tensorboard_log',
                        help="Path for the tensorboard log")
    parser.add_argument('-st', action='store', dest='sequence_type',
                        help="Type of demand sequence to use")
    return parser


def args_from_config(path: str) -> Dict:
    """Loads YAML config file in place of arguments. YAML has special parameter:
    'parents' which causes parent files to be read with options overridden by
    this one. Takes list of paths, priority for arguments is last set seen."""
    with open(path, 'r') as stream:
        config = yaml.safe_load(stream)
    if 'parents' in config:
        for parent in config['parents']:
            parent_config = args_from_config(parent)
            config = {**parent_config, **config}
    return config


def read_hyperparameters(args: Dict) -> Dict:
    """Reads in hyperparameters from json config file"""
    hyperparameters = {}
    if args['hyperparameters']:
        with open(args['hyperparameters'], 'r') as stream:
            hyperparameters = json.load(stream)
    return hyperparameters


def seed(seed: int):
    """Attempts to seed things for ability to get deterministic results"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def run_training(config: Dict):
    # TODO: give this a bit of a cleanup
    """Runs training based on config passed in"""
    multiprocessing.set_start_method('spawn')

    print("Run configuration:")
    print(config)
    seed(config['seed'])

    # read config
    hyperparameters = read_hyperparameters(config)
    graphs = graphs_from_args(config['graphs'])
    policy, policy_kwargs = policy_from_args(config, graphs)
    demands = demands_from_args(config, graphs)
    env_kwargs = env_kwargs_from_args(config)
    env_name = config['env_name']
    timesteps = config['timesteps']
    parallelism = config['parallelism']
    log_name = config['log_name']
    replay_steps = config['replay_steps']
    model_name = config['model_name']
    tensorboard_log = config['tensorboard_log']

    oblivious_routings = None

    # make env
    env = lambda: gym.make(env_name,
                           dm_sequence=demands,
                           graphs=graphs,
                           oblivious_routings=oblivious_routings,
                           **env_kwargs)
    vec_env = SubprocVecEnv([env for _ in range(parallelism)],
                            start_method="spawn")

    # make model
    model = PPO2(policy,
                 vec_env,
                 cliprange_vf=-1,
                 verbose=1,
                 policy_kwargs=policy_kwargs,
                 tensorboard_log=tensorboard_log,
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
            action, state = model.predict(obs, state=state,
                                          deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            print(reward)
            print(info)
            if sum(info[0]['edge_set']) == 0:
                reward_inc += 1
                total_rewards += info[0]['real_reward']
        print("Mean reward: ", total_rewards / reward_inc)
    else:
        for i in range(replay_steps - 1):
            action, state = model.predict(obs, state=state,
                                          deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            print(reward)
            print(info)
            total_rewards += reward[0]
        print("Mean reward: ", total_rewards / (replay_steps - 1))
    vec_env.close()


def run_model(config: Dict):
    # TODO: give this a bit of a cleanup
    """Runs test on a model based on config passed in"""
    multiprocessing.set_start_method('spawn')

    print("Run configuration:")
    print(config)
    seed(config['seed'])

    # read config
    graphs = graphs_from_args(config['graphs'])
    policy_name = config['policy_name']
    model_path = config['model_path']
    demands = demands_from_args(config, graphs)
    env_kwargs = env_kwargs_from_args(config)
    env_name = config['env_name']
    parallelism = config['parallelism']
    replay_steps = config['replay_steps']

    oblivious_routings = [routing_baselines.shortest_path_routing(graph) for
                          graph in graphs]

    # make env
    env = lambda: gym.make(env_name,
                           dm_sequence=demands,
                           graphs=graphs,
                           oblivious_routings=oblivious_routings,
                           **env_kwargs)

    if policy_name == 'lstm':
        envs = DummyVecEnv([env] * parallelism)
    else:
        envs = DummyVecEnv([env])

    # load
    model = PPO2.load(model_path)

    # execute
    obs = envs.reset()
    state = None
    utilisations = []
    opt_utilisations = []
    oblivious_utilisations = []
    if env_name == 'ddr-iterative-v0':
        for i in range(replay_steps - 1):
            action, state = model.predict(obs, state=state, deterministic=True)
            obs, reward, done, info = envs.step(action)
            if info[0]['iter_idx'] == 0:
                utilisations.append(info[0]['utilisation'])
                opt_utilisations.append(info[0]['opt_utilisation'])
                oblivious_utilisations.append(info[0]['oblivious_utilisation'])
    else:
        for i in range(replay_steps - 1):
            action, state = model.predict(obs, state=state, deterministic=True)
            obs, reward, done, info = envs.step(action)
            utilisations.append(info[0]['utilisation'])
            opt_utilisations.append(info[0]['opt_utilisation'])
            oblivious_utilisations.append(info[0]['oblivious_utilisation'])
    envs.close()

    # write the results to file
    result = {"utilisations": utilisations,
              "opt_utilisations": opt_utilisations,
              "oblivious_utilisations": oblivious_utilisations}
    if 'output_path' in config:
        data = {**config, **result}
        with jsonlines.open(config['output_path'], 'w') as f:
            f.write(data)
