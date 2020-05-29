import multiprocessing
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import gym
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
import jsonlines

from ddr_learning_helpers import routing_baselines
from ddr_learning_helpers.runs import *


def run_model(env_name: str, graphs: List[nx.DiGraph],
              demands: List[List[List[Tuple[np.ndarray, float]]]], model_path: str,
              replay_steps: int = 10,
              env_kwargs: Dict = {},
              parallelism: int = 4,
              policy_name: str = None):
    oblivious_routings = [routing_baselines.shortest_path_routing(graph) for graph in graphs]

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
        replay_steps = replay_steps * envs.envs[0].graphs[
            envs.envs[0].graph_index].number_of_edges()
        for i in range(replay_steps - 1):
            action, state = model.predict(obs, state=state, deterministic=True)
            obs, reward, done, info = envs.step(action)
            print(reward)
            print(info)
            if info[0]['iter_idx'] == 0:
                utilisations.append(info[0]['utilisation'])
                opt_utilisations.append(info[0]['opt_utilisation'])
                oblivious_utilisations.append(info[0]['oblivious_utilisation'])
    else:
        for i in range(replay_steps - 1):
            action, state = model.predict(obs, state=state, deterministic=True)
            obs, reward, done, info = envs.step(action)
            print(reward)
            print(action)
            print(info)
            utilisations.append(info[0]['utilisation'])
            opt_utilisations.append(info[0]['opt_utilisation'])
            oblivious_utilisations.append(info[0]['oblivious_utilisation'])
    print("Mean reward: ", np.mean(np.divide(utilisations, opt_utilisations)))
    print("Mean oblivious reward: ",
          np.mean(np.divide(oblivious_utilisations, opt_utilisations)))
    envs.close()

    return utilisations, opt_utilisations, oblivious_utilisations


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

    result = run_model(args['env'], graphs, demands, args['model_path'],
                       args['replay_steps'], env_kwargs,
                       args['parallelism'], args['policy'])

    if 'output_path' in args:
        data = {**args, **result}
        with jsonlines.open(args['output_path'], 'w') as f:
            f.write(data)
