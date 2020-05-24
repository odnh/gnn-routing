"""
A selection of functions for help in running experiments, tuning, models, and
training rom the command line.
"""
import argparse
import json
from typing import List, Dict, Tuple
import os

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
        demands_with_opt = [[(demand, mlu.opt(demand)) for demand in sequence] for
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
