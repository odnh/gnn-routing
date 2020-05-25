"""
Script to run experiments. All configuration is in code because its much easier
and less effort than the yaml hierarchy I was using before.
"""
import multiprocessing

from ddr_learning_helpers.runs import run_training, run_model

# base definitions for each policy type
base = {
    'mlp': {'env_name': 'ddr-softmin-v0',
            'policy': 'mlp',
            'hyperparameters': 'configs/hpm.json'},
    'gnn': {'env_name': 'ddr-softmin-v0',
            'policy': 'gnn',
            'gnn_iterations': 2,
            'vf_arch': 'mlp',
            'hyperparameters': 'configs/hpg.json'},
    'iter': {'env_name': 'ddr-iterative-v0',
             'policy': 'iter',
             'gnn_iterations': 2,
             'vf_arch': 'mlp',
             'hyperparameters': 'configs/hpi.json'},
    'lstm': {'env_name': 'ddr-softmin-v0',
             'policy': 'lstm',
             'hyperparameters': 'configs/hpl.json',
             'memory_length': 1},
}

# base definitions for each training
train = {
    # exp 1
    '1':
        {
            'memory_length': 1,
            'graphs': ['Abilene'],
            'timesteps': 500000,
            'sequence_type': 'cyclical',
            'demand_seeds': [2],
            'cycle_length': 1,
            'sparsity': 0.0,
            'sequence_length': 2,
            'seed': 1,
            'parallelism': 8,
            'tensorboard_log': None,
            'log_name': 1,
            'replay_steps': 0
        },
    # exp 2
    '2':
        {
            'memory_length': 10,
            'graphs': ['Abilene'],
            'timesteps': 500000,
            'sequence_type': 'cyclical',
            'demand_seeds': [1, 2, 3, 4, 5],
            'cycle_length': 5,
            'sparsity': 0.0,
            'sequence_length': 6,
            'seed': 1,
            'parallelism': 8,
            'tensorboard_log': None,
            'log_name': 2,
            'replay_steps': 0
        },
    # exp 3
    '3':
        {
            'memory_length': 10,
            'graphs': ["Abilene", "Abilene:e+:1", "Abilene:e-:1",
                       "Abilene:n+:1", "Abilene:n-:1"],
            'timesteps': 500000,
            'sequence_type': 'cyclical',
            'demand_seeds': [1, 2, 3, 4, 5],
            'cycle_length': 5,
            'sparsity': 0.0,
            'sequence_length': 6,
            'seed': 1,
            'parallelism': 8,
            'tensorboard_log': None,
            'log_name': 3,
            'replay_steps': 0
        },
    # exp 4 TODO: add options to use real dataset and fill in here
    '4':
        {
            'memory_length': 10,
            'graphs': ["Abilene"],
            'timesteps': 500000,
            'sequence_type': 'cyclical',
            'demand_seeds': [1, 2, 3, 4, 5],
            'cycle_length': 5,
            'sparsity': 0.0,
            'sequence_length': 6,
            'seed': 1,
            'parallelism': 8,
            'tensorboard_log': None,
            'log_name': 4,
            'replay_steps': 0
        }
}

# base definitions for tests TODO: actually fill these in
test = {
    '1': {
        '1': {
            'replay_steps': 2
        },
        '2': {
            'demand_seeds': [1],
            'cycle_length': 10,
            'sequence_length': 11,
            'replay_steps': 11
        }
        # TODO: also train on multiple DMs but single cycle single history
    },
    '2': {
        '1': {
            'replay_steps': 6
        },
        '2': {
            'demand_seeds': [6],
            'cycle_length': 5,
            'sequence_length': 6,
            'replay_steps': 6
        }
        # TODO: may need to extend this with more examples
    },
    '3': {
        '1': {
            'replay_steps': 6
        },
        '2': {
            'graphs': 'Abilene:e:-:2',
            'replay_steps': 6
        },
        '3': {
            'graphs': 'Abilene:e:+:2',
            'replay_steps': 6
        },
        '4': {
            'graphs': 'Abilene:n:-:2',
            'replay_steps': 6
        },
        '5': {
            'graphs': 'Abilene:n:+:2',
            'replay_steps': 6
        },
        '6': {
            'graphs': 'BtEurope',
            'replay_steps': 6
        }
    },
    '4': {
        '1': {}
    }
}


# functions to train and run
def run_experiment(spec_id: str, policy_id: str, test_id: str):
    """Run a specific experiment. Model must already be trained"""
    model_path  = "models/{}-{}"    .format(spec_id, policy_id)
    output_path = "results/{}.{}-{}".format(spec_id, test_id, policy_id)
    config = {**train[spec_id],
              **test[spec_id][test_id],
              **base[policy_id],
              'model_name': model_path,
              'output_path': output_path}
    run_model(config)


def train_model(spec_id: str, policy_id: str):
    """Train a specific model"""
    model_path = "models/{}-{}".format(spec_id, policy_id)
    config = {**train[spec_id],
              **base[policy_id],
              'model_name': model_path}
    run_training(config)


def train_models():
    """Train all models"""
    for spec_id in train.keys():
        for policy_id in base.keys():
            train_model(spec_id, policy_id)


def run_experiments():
    """Run all experiments"""
    for spec_id in train.keys():
        for policy_id in base.keys():
            if spec_id == '3' and policy_id not in ['gnn', 'iter']:
                # graph generalisation not work for other policies
                continue
            for test_id in test[spec_id].keys():
                run_experiment(spec_id, policy_id, test_id)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    train_models()
    run_experiments()
