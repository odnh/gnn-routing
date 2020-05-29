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
            'layer_size': 64,
            'layer_count': 3,
            'gnn_iterations': 2,
            'vf_arch': 'shared',
            'hyperparameters': 'configs/hpg.json'},
    'iter': {'env_name': 'ddr-iterative-v0',
             'policy': 'iter',
             'layer_size': 64,
             'layer_count': 3,
             'gnn_iterations': 2,
             'vf_arch': 'shared',
             'hyperparameters': 'configs/hpi.json'},
    'lstm': {'env_name': 'ddr-softmin-v0',
             'policy': 'lstm',
             'hyperparameters': 'configs/hpl.json',
             'memory_length': 1},
}

# base definitions for each training
train = {
    '1.1':
        {
            'memory_length': 1,
            'graphs': ['basic2'],
            'timesteps': 100000,
            'demand_type': 'bimodal',
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
    '1.2':
        {
            'memory_length': 1,
            'graphs': ['basic2'],
            'timesteps': 100000,
            'demand_type': 'bimodal',
            'sequence_type': 'cyclical',
            'demand_seeds': list(range(10000)),
            'cycle_length': 101,
            'sparsity': 0.0,
            'sequence_length': 101,
            'seed': 1,
            'parallelism': 8,
            'tensorboard_log': None,
            'log_name': 1,
            'replay_steps': 0
        },
    '2.1':
        {
            'memory_length': 10,
            'graphs': ['basic2'],
            'timesteps': 100000,
            'demand_type': 'bimodal',
            'sequence_type': 'cyclical',
            'demand_seeds': list(range(10000)),
            'cycle_length': 5,
            'sparsity': 0.0,
            'sequence_length': 6,
            'seed': 1,
            'parallelism': 8,
            'tensorboard_log': None,
            'log_name': 2,
            'replay_steps': 0
        },
    '2.2':
        {
            'memory_length': 10,
            'graphs': ['basic2'],
            'timesteps': 100000,
            'demand_type': 'gravity',
            'sequence_type': 'cyclical',
            'demand_seeds': list(range(10000)),
            'cycle_length': 5,
            'sparsity': 0.5,
            'sequence_length': 6,
            'seed': 1,
            'parallelism': 8,
            'tensorboard_log': None,
            'log_name': 2,
            'replay_steps': 0
        },
    '2.3':
        {
            'memory_length': 10,
            'graphs': ['basic2'],
            'timesteps': 100000,
            'demand_type': 'bimodal',
            'sequence_type': 'average',
            'demand_seeds': list(range(10000)),
            'cycle_length': 5,
            'sparsity': 0.0,
            'sequence_length': 6,
            'seed': 1,
            'parallelism': 8,
            'tensorboard_log': None,
            'log_name': 2,
            'replay_steps': 0
        },
    '2.4':
        {
            'memory_length': 10,
            'graphs': ['basic2'],
            'timesteps': 100000,
            'demand_type': 'gravity',
            'sequence_type': 'average',
            'demand_seeds': list(range(10000)),
            'cycle_length': 5,
            'sparsity': 0.5,
            'sequence_length': 6,
            'seed': 1,
            'parallelism': 8,
            'tensorboard_log': None,
            'log_name': 2,
            'replay_steps': 0
        },
    '3.1':
        {
            'memory_length': 10,
            'graphs': ['basic2'],
            'timesteps': 100000,
            'demand_type': 'bimodal',
            'sequence_type': 'cyclical',
            'demand_seeds': [1, 1, 1, 1, 1],
            'demand_qs': [1, 2, 3, 4, 5],
            'sparsity': 0.0,
            'sequence_length': 6,
            'seed': 1,
            'parallelism': 8,
            'tensorboard_log': None,
            'log_name': 2,
            'replay_steps': 0
        },
    '4.1':
        {
            'memory_length': 10,
            'graphs': ["basic2", "basic2:e:+:1", "basic2:e:-:1",
                       "basic2:n:+:1", "basic2:n:-:1", "basic2:e:+:2",
                       "basic2:e:-:2", "basic2:n:+:2", "basic2:n:-:2"],
            'graph_indices': [0, 1, 2, 3, 4],
            'timesteps': 100000,
            'demand_type': 'bimodal',
            'sequence_type': 'cyclical',
            'demand_seeds': list(range(10000)),
            'cycle_length': 5,
            'sparsity': 0.0,
            'sequence_length': 6,
            'seed': 1,
            'parallelism': 8,
            'tensorboard_log': None,
            'log_name': 3,
            'replay_steps': 0
        },
    '4.2':
        {
            'memory_length': 10,
            'graphs': ["basic2", "basic", "full", "Abilene"],
            'graph_indices': [0, 1],
            'timesteps': 100000,
            'demand_type': 'bimodal',
            'sequence_type': 'cyclical',
            'demand_seeds': list(range(10000)),
            'cycle_length': 5,
            'sparsity': 0.0,
            'sequence_length': 6,
            'seed': 1,
            'parallelism': 8,
            'tensorboard_log': None,
            'log_name': 3,
            'replay_steps': 0
        },
    '5.1':
        {
            'memory_length': 10,
            'graphs': ["totem"],
            'timesteps': 100000,
            'demand_type': 'bimodal',  # this arg is required but ignored
            'sequence_type': 'totem',
            'demand_seeds': [1, 2, 3, 4, 5],
            'cycle_length': 0,
            'sparsity': 0.0,
            'sequence_length': 6,
            'seed': 1,
            'parallelism': 8,
            'tensorboard_log': None,
            'log_name': 4,
            'replay_steps': 0
        }
}

# base definitions for tests
test = {
    '1.1':
        {
            'demand_seeds': [2],
            'replay_steps': 2
        },
    '1.2':
        {
            'demand_seeds': list(range(10000, 20000)),
            'replay_steps': 101
        },
    '2.1':
        {
            'demand_seeds': list(range(10000, 20000)),
            'replay_steps': 6
        },
    '2.2':
        {
            'demand_seeds': list(range(10000, 20000)),
            'replay_steps': 6
        },
    '2.3':
        {
            'demand_seeds': list(range(10000, 20000)),
            'replay_steps': 6
        },
    '2.4':
        {
            'demand_seeds': list(range(10000, 20000)),
            'replay_steps': 6
        },
    '3.1':
        {
            'demand_seeds': [2, 2, 2, 2, 2],
            'demand_qs': [1, 2, 3, 4, 5],
            'replay_steps': 6
        },
    '4.1':
        {
            'graphs': ["basic2", "basic2:e:+:1", "basic2:e:-:1",
                       "basic2:n:+:1", "basic2:n:-:1", "basic2:e:+:2",
                       "basic2:e:-:2", "basic2:n:+:2", "basic2:n:-:2"],
            'graph_indices': [5, 6, 7, 8],
            'replay_steps': 6
        },
    '4.2':
        {
            'graphs': ["basic2", "basic", "full", "Abilene"],
            'graph_indices': [2, 3],
            'replay_steps': 6
        },
    '5.1':
        {
            'demand_seeds': list(range(50, 100)),
            'replay_steps': 5
        }
}


# functions to train and run
def run_experiment(spec_id: str, policy_id: str):
    """Run a specific experiment. Model must already be trained"""
    model_path = "models/{}-{}".format(spec_id, policy_id)
    output_path = "results/{}-{}".format(spec_id, policy_id)
    config = {**train[spec_id],
              **test[spec_id],
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
            if spec_id in ['4.1', '4.2'] and policy_id not in ['gnn', 'iter']:
                # graph generalisation not work for other policies
                continue
            train_model(spec_id, policy_id)


def run_experiments():
    """Run all experiments"""
    for spec_id in test.keys():
        for policy_id in base.keys():
            if spec_id in ['4.1', '4.2'] and policy_id not in ['gnn', 'iter']:
                # graph generalisation not work for other policies
                continue
            run_experiment(spec_id, policy_id)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    train_models()
    run_experiments()
