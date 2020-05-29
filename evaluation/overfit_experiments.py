"""
Script to run experiments which assess where overfitting occurs.
"""
import multiprocessing

from ddr_learning_helpers.runs import run_training, run_model

# base definitions for each policy type
policy_conf = {
    'mlp': {'env_name': 'ddr-softmin-v0',
            'policy': 'mlp',
            'hyperparameters': 'configs/hpm.json'},
    'gnn': {'env_name': 'ddr-softmin-v0',
            'policy': 'gnn',
            'layer_size': 64,
            'layer_count': 3,
            'gnn_iterations': 2,
            'vf_arch': 'shared',
            'hyperparameters': 'configs/hpg.json'},
}

base_conf = {
    'memory_length': 1,
    'graphs': ['basic2'],
    'timesteps': 100000,
    'demand_type': 'bimodal',
    'sequence_type': 'cyclical',
    'cycle_length': 1,
    'sparsity': 0.0,
    'sequence_length': 2,
    'seed': 1,
    'parallelism': 4,
    'tensorboard_log': None,
    'log_name': 1,
    'replay_steps': 2
}


exp_conf = {
    '1':
        {
            'demand_seeds': [1],
        },
    '2':
        {
            'demand_seeds': [2],
        },
    '3':
        {
            'demand_seeds': [3],
        },
    '4':
        {
            'demand_seeds': [4],
        },
    '5':
        {
            'demand_seeds': [5],
        },
    '1_2':
        {
            'demand_seeds': [1, 2],
        },
    '1_3':
        {
            'demand_seeds': [1, 2, 3],
        },
    '1_4':
        {
            'demand_seeds': [1, 2, 3, 4],
        },
    '1_5':
        {
            'demand_seeds': [1, 2, 3, 4, 5],
        },
    'out':
        {
            'demand_seeds': [1],
            'cycle_length': 100,
            'sequence_length': 101,
            'replay_steps': 101
        }
}


# functions to train and run
def run_experiment(model_id: str, test_id: str, policy_id: str):
    """Run a specific experiment. Model must already be trained"""
    model_path = "models/overfit-{}-{}".format(model_id, policy_id)
    output_path = "results/overfit-{}-{}-{}".format(model_id, test_id, policy_id)
    config = {**base_conf,
              **exp_conf[test_id],
              **policy_conf[policy_id],
              'model_name': model_path,
              'output_path': output_path}
    run_model(config)


def train_model(spec_id: str, policy_id: str):
    """Train a specific model"""
    model_path = "models/overfit-{}-{}".format(spec_id, policy_id)
    config = {**base_conf,
              **exp_conf[spec_id],
              **policy_conf[policy_id],
              'model_name': model_path}
    run_training(config)


def train_models():
    """Train all models"""
    for spec_id in exp_conf.keys():
        if spec_id != 'out':
            train_model(spec_id, 'mlp')


def run_experiments():
    """Run all experiments"""
    for model_id, tests in [
        ('1', ['1', 'out']),
        ('1_2', ['1', '2', 'out']),
        ('1_3', ['1', '2', '3', 'out']),
        ('1_4', ['1', '2', '3', '4', 'out']),
        ('1_5', ['1', '2', '3', '4', '5', 'out'])]:
        for test_id in tests:
            run_experiment(model_id, test_id, 'mlp')


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    train_models()
    run_experiments()
