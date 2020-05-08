import os
from typing import List, Dict, Tuple

import networkx as nx
import yaml
from ddr_learning_helpers import graphs

import warnings
warnings.simplefilter("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import gym
import gym_ddr.envs.demand_matrices as dm
from gym_ddr.envs.max_link_utilisation import MaxLinkUtilisation
import numpy as np
from stable_baselines import PPO2
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines_ddr.gnn_policy import MlpDdrPolicy, MlpLstmDdrPolicy, \
    GnnDdrPolicy, GnnDdrIterativePolicy, GnnLstmDdrPolicy, \
    GnnLstmDdrIterativePolicy


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
                   log_name: str = "") -> float:
    oblivious_routing = None  # yates.get_oblivious_routing(graph)

    # make env
    env = lambda: gym.make(env_name,
                           dm_sequence=demands,
                           graph=graph,
                           oblivious_routing=oblivious_routing,
                           **env_kwargs)
    vec_env = SubprocVecEnv([env for _ in range(parallelism)],
                            start_method="spawn")

    hyperparameters['nminibatches'] = max(
        int(hyperparameters['n_steps'] / hyperparameters['batch_size']), 1)
    del hyperparameters['batch_size']

    # make model
    model = PPO2(policy,
                 vec_env,
                 cliprange_vf=-1,
                 verbose=0,
                 policy_kwargs=policy_kwargs,
                 tensorboard_log="./tune_tensorboard/",
                 **hyperparameters)

    # learn
    if env_name == 'ddr-iterative-v0':
        model.learn(total_timesteps=timesteps, tb_log_name=log_name,
                    callback=true_reward_callback)
        evaluation_steps = len(demands[0]) * graph.number_of_edges()
    else:
        model.learn(total_timesteps=timesteps, tb_log_name=log_name)
        evaluation_steps = len(demands[0])

    # see how good this model is
    if env_name == 'ddr-iterative-v0':
        obs = vec_env.reset()
        reward_inc = 0
        total_rewards = 0
        for i in range(evaluation_steps):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = vec_env.step(action)
            print(rewards)
            print(info[0])
            if sum(info[0]['edge_set']) == 0:
                reward_inc += 1
                total_rewards += info[0]['real_reward'] 
    else:
        obs = vec_env.reset()
        total_rewards = 0
        for i in range(evaluation_steps):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = vec_env.step(action)
            print(rewards)
            print(info[0])
            total_rewards += sum(rewards)
    vec_env.close()
    return total_rewards


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
    if args['memory_length']:
        env_kwargs['dm_memory_length'] = args['memory_length']
    return env_kwargs


def run_tuning(hyperparameters: Dict, config_path: str):
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    print("Run configuration:")
    print(config)

    graph = graph_from_args(config)
    policy, policy_kwargs = policy_from_args(config, graph)
    demands = demands_from_args(config, graph)
    env_kwargs = env_kwargs_from_args(config)

    return run_experiment(config['env'], policy, graph, demands, env_kwargs,
                   policy_kwargs, hyperparameters,
                   config['timesteps'], config['parallelism'])
