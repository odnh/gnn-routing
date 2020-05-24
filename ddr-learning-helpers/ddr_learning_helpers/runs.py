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
from ddr_learning_helpers import graphs
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


def demands_from_args(args: Dict, graph: nx.DiGraph) -> List[
    List[Tuple[np.ndarray, float]]]:
    """Uses program agruments to build demand sequnces"""
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
    """Uses program arguments to build policy network"""
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
    """Uses program arguments to build graph"""
    if args['graph']:
        graph = graphs.topologyzoo(args['graph'], 10000)
    else:
        graph = graphs.basic()
    return graph


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
                        help="Config file to read. Other options override this")
    parser.add_argument('-hy', action='store', dest='hyperparameters',
                        help="Hyperprameter (json) config file to read.")
    parser.add_argument('-e', action='store', dest='env',
                        help="Name of environment to train on")
    parser.add_argument('-p', action='store', dest='policy',
                        help="Name of the policy to use")
    parser.add_argument('-g', action='store', dest='graph',
                        help="Name of graph to train on")
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
    return parser


def args_from_config(args: Dict) -> Dict:
    """Loads yml config file in place of arguments"""
    with open(args['config'], 'r') as stream:
        config = yaml.safe_load(stream)
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
