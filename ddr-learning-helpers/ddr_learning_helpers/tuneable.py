"""
An implementation of an experiment runner that can be called with various
different hyperparameters by OpenTuner
"""
import os

import warnings

warnings.simplefilter("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from .runs import *


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
            if sum(info[0]['edge_set']) == 0:
                reward_inc += 1
                total_rewards += info[0]['real_reward']
    else:
        obs = vec_env.reset()
        total_rewards = 0
        for i in range(evaluation_steps):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = vec_env.step(action)
            total_rewards += sum(rewards)
    vec_env.close()
    return total_rewards


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
