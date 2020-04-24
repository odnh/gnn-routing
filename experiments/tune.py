import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import os
import time
import gym
import gym_ddr.envs.demand_matrices as dm
from gym_ddr.envs.max_link_utilisation import MaxLinkUtilisation
import numpy as np
from ddr_learning_helpers import graphs, yates
from stable_baselines.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv, VecEnv
from stable_baselines.common.callbacks import BaseCallback, EvalCallback
from stable_baselines import PPO2
from stable_baselines_ddr.gnn_policy import GnnDdrPolicy

import optuna
from optuna.pruners import SuccessiveHalvingPruner, MedianPruner
from optuna.samplers import RandomSampler, TPESampler
from optuna.integration.skopt import SkoptSampler


################################################################################
# CONFIG VALUES
################################################################################

save_path = "/local/scratch/oh260/tune/"
hyperparams = {}
env_kwargs = {}
log_folder = "/local/scratch/oh260/tune/"
n_trials = 10
n_timesteps = -1
seed = 42
n_jobs = 4
n_envs = 4
sampler = "random"
pruner = "halving"
verbose = True

################################################################################
# SET UP GRAPH & DMS
################################################################################

# load/generate graph
# graph = graphs.topologyzoo("TLex", 10000)
graph = graphs.basic()

## ENV PARAMETERS
rs = np.random.RandomState()  # Random state
dm_memory_length = 10  # Length of memory of dms in each observation
num_demands = graph.number_of_nodes() * (
            graph.number_of_nodes() - 1)  # Demand matrix size (dependent on graph size
dm_generator_getter = lambda seed: dm.cyclical_sequence(
    # A function that returns a generator for a sequence of demands
    lambda rs_l: dm.bimodal_demand(num_demands, rs_l), 50, 5, 0.0, seed=seed)
# demand_sequences = [list(dm_generator_getter()) for i in range(2)]  # Collect the generator into a sequence
mlu = MaxLinkUtilisation(graph)  # Friendly max link utilisation class
demand_sequences = map(dm_generator_getter, [32, 32])
demands_with_opt = [[(demand, mlu.opt(demand)) for demand in sequence] for
                    # Merge opt calculations into the demand sequence
                    sequence in demand_sequences]

oblivious_routing = None  # yates.get_oblivious_routing(graph)

# make env
env = lambda: gym.make('ddr-softmin-v0',
                       dm_sequence=demands_with_opt,
                       dm_memory_length=dm_memory_length,
                       graph=graph,
                       oblivious_routing=oblivious_routing)

################################################################################
# HYPERPARAMETER TUNE FUNCTIONS
################################################################################

class TrialEvalCallback(EvalCallback):
    """
    Callback used for evaluating and reporting a trial.
    """
    def __init__(self, eval_env, trial, n_eval_episodes=5,
                 eval_freq=10000, deterministic=True, verbose=0):

        super(TrialEvalCallback, self).__init__(eval_env=eval_env, n_eval_episodes=n_eval_episodes,
                                                eval_freq=eval_freq,
                                                deterministic=deterministic,
                                                verbose=verbose)
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super(TrialEvalCallback, self)._on_step()
            self.eval_idx += 1
            # report best or report current ?
            # report num_timesteps or elasped time ?
            self.trial.report(-1 * self.last_mean_reward, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune(self.eval_idx):
                self.is_pruned = True
                return False
        return True

def create_env(n_envs, eval_env=False):
    """
    Create the environment and wrap it if necessary
    :param n_envs: (int)
    :param eval_env: (bool) Whether is it an environment used for evaluation or not
    :return: (Union[gym.Env, VecEnv])
    :return: (gym.Env)
    """
    global hyperparams
    global env_kwargs

    # Do not log eval env (issue with writing the same file)
    log_dir = None if eval_env else save_path

    if n_envs == 1:
        vec_env = DummyVecEnv([make_env])
    else:
        vec_env = SubprocVecEnv([make_env for _ in range(n_envs)])

    normalised_env = VecNormalize(vec_env, training=True, norm_obs=True, norm_reward=False)
    return normalised_env

def create_model(*_args, **kwargs):
    """
    Helper to create a model with different hyperparameters
    """
    return PPO2(GnnDdrPolicy, env=create_env(n_envs), policy_kwargs={'network_graph': graph,
                                'dm_memory_length': dm_memory_length,
                                'vf_arch': "graph"}, tensorboard_log="./hyperparam_tensorboard", **kwargs)


def hyperparam_optimization(model_fn, env_fn, n_trials=10, n_timesteps=5000, hyperparams=None,
                            n_jobs=1, sampler_method='random', pruner_method='halving',
                            seed=0, verbose=1):
    """
    :param algo: (str)
    :param model_fn: (func) function that is used to instantiate the model
    :param env_fn: (func) function that is used to instantiate the env
    :param n_trials: (int) maximum number of trials for finding the best hyperparams
    :param n_timesteps: (int) maximum number of timesteps per trial
    :param hyperparams: (dict)
    :param n_jobs: (int) number of parallel jobs
    :param sampler_method: (str)
    :param pruner_method: (str)
    :param seed: (int)
    :param verbose: (int)
    :return: (pd.Dataframe) detailed result of the optimization
    """
    # TODO: eval each hyperparams several times to account for noisy evaluation
    # TODO: take into account the normalization (also for the test env -> sync obs_rms)
    if hyperparams is None:
        hyperparams = {}

    n_startup_trials = 10
    # test during 5 episodes
    n_eval_episodes = 5
    # evaluate every 20th of the maximum budget per iteration
    n_evaluations = 20
    eval_freq = int(n_timesteps / n_evaluations)

    # n_warmup_steps: Disable pruner until the trial reaches the given number of step.
    if sampler_method == 'random':
        sampler = RandomSampler(seed=seed)
    elif sampler_method == 'tpe':
        sampler = TPESampler(n_startup_trials=n_startup_trials, seed=seed)
    elif sampler_method == 'skopt':
        # cf https://scikit-optimize.github.io/#skopt.Optimizer
        # GP: gaussian process
        # Gradient boosted regression: GBRT
        sampler = SkoptSampler(skopt_kwargs={'base_estimator': "GP", 'acq_func': 'gp_hedge'})
    else:
        raise ValueError('Unknown sampler: {}'.format(sampler_method))

    if pruner_method == 'halving':
        pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=4, min_early_stopping_rate=0)
    elif pruner_method == 'median':
        pruner = MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=n_evaluations // 3)
    elif pruner_method == 'none':
        # Do not prune
        pruner = MedianPruner(n_startup_trials=n_trials, n_warmup_steps=n_evaluations)
    else:
        raise ValueError('Unknown pruner: {}'.format(pruner_method))

    if verbose > 0:
        print("Sampler: {} - Pruner: {}".format(sampler_method, pruner_method))

    study = optuna.create_study(sampler=sampler, pruner=pruner)
    algo_sampler = sample_ppo2_params

    def objective(trial):

        kwargs = hyperparams.copy()

        trial.model_class = None
        kwargs.update(algo_sampler(trial))

        model = model_fn(**kwargs)

        eval_env = env_fn(n_envs=1, eval_env=True)
        # Account for parallel envs
        eval_freq_ = eval_freq
        if isinstance(model.get_env(), VecEnv):
            eval_freq_ = max(eval_freq // model.get_env().num_envs, 1)
        eval_callback = TrialEvalCallback(eval_env, trial, n_eval_episodes=n_eval_episodes,
                                          eval_freq=eval_freq_, deterministic=True)

        try:
            model.learn(n_timesteps, callback=eval_callback)
            # Free memory
            model.env.close()
            eval_env.close()
        except AssertionError:
            # Sometimes, random hyperparams can generate NaN
            # Free memory
            model.env.close()
            eval_env.close()
            raise optuna.exceptions.TrialPruned()
        is_pruned = eval_callback.is_pruned
        cost = -1 * eval_callback.last_mean_reward

        del model.env, eval_env
        del model

        if is_pruned:
            raise optuna.exceptions.TrialPruned()

        return cost

    try:
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    except KeyboardInterrupt:
        pass

    print('Number of finished trials: ', len(study.trials))

    print('Best trial:')
    trial = study.best_trial

    print('Value: ', trial.value)

    print('Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    return study.trials_dataframe()


def sample_ppo2_params(trial):
    """
    Sampler for PPO2 hyperparams.
    :param trial: (optuna.trial)
    :return: (dict)
    """
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    n_steps = trial.suggest_categorical('n_steps', [16, 32, 64, 128, 256, 512, 1024, 2048])
    gamma = trial.suggest_categorical('gamma', [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform('lr', 1e-5, 1)
    ent_coef = trial.suggest_loguniform('ent_coef', 0.00000001, 0.1)
    cliprange = trial.suggest_categorical('cliprange', [0.1, 0.2, 0.3, 0.4])
    noptepochs = trial.suggest_categorical('noptepochs', [1, 5, 10, 20, 30, 50])
    lam = trial.suggest_categorical('lamdba', [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])

    if n_steps < batch_size:
        nminibatches = 1
    else:
        nminibatches = int(n_steps / batch_size)

    return {
        'n_steps': n_steps,
        'nminibatches': nminibatches,
        'gamma': gamma,
        'learning_rate': learning_rate,
        'ent_coef': ent_coef,
        'cliprange': cliprange,
        'noptepochs': noptepochs,
        'lam': lam
    }


################################################################################
# RUN TUNING
################################################################################

if __name__ == "__main__":
    data_frame = hyperparam_optimization(create_model, create_env, n_trials=n_trials,
                                         n_timesteps=n_timesteps, hyperparams=hyperparams,
                                         n_jobs=n_jobs, seed=seed,
                                         sampler_method=sampler, pruner_method=pruner,
                                         verbose=verbose)

    report_name = "report_{}-trials-{}-{}-{}_{}.csv".format(n_trials, n_timesteps,
                                                            sampler, pruner, int(time.time()))

    log_path = os.path.join(log_folder, report_name)

    print("Writing report to {}".format(log_path))

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    data_frame.to_csv(log_path)
