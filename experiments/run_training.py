import multiprocessing
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import gym
from stable_baselines import PPO2
from stable_baselines.common.vec_env import SubprocVecEnv

from ddr_learning_helpers.runs import *


def run_experiment(env_name: str, policy: ActorCriticPolicy, graphs: List[nx.DiGraph],
                   demands: List[List[Tuple[np.ndarray, float]]],
                   env_kwargs: Dict = {}, policy_kwargs: Dict = {},
                   hyperparameters: Dict = {}, timesteps: int = 10000,
                   parallelism: int = 4, model_name: str = "",
                   log_name: str = "", replay_steps: int = 10,
                   tensorboard_log: str = None):
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


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    parser = argparser()
    cli_args = vars(parser.parse_args())

    if 'config' in cli_args:
        args = args_from_config(cli_args['config'])
        # and add cli overrides
        args.update((k, cli_args[k]) for k in cli_args.keys() if
                    cli_args[k] is not None)
    else:
        args = cli_args

    print("Run configuration:")
    print(args)
    seed(args['seed'])

    hyperparameters = read_hyperparameters(args)
    graphs = graphs_from_args(args['graphs'])
    policy, policy_kwargs = policy_from_args(args, graphs)
    demands = demands_from_args(args, graphs)
    env_kwargs = env_kwargs_from_args(args)

    run_experiment(args['env'], policy, graphs, demands, env_kwargs,
                   policy_kwargs, hyperparameters, args['timesteps'],
                   args['parallelism'], args['log_name'], args['model_name'],
                   args['replay_steps'], args['tensorboard_log'])
