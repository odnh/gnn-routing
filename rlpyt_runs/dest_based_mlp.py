"""
Train DDR for destination-based routing using PPO
"""
import gym_gnn.envs.demand_matrices as dm

import numpy as np
import networkx as nx

### RLPYT HELPER BITS AND PIECES


### NETWORK HELPER BITS AND PIECES

graph = nx.DiGraph()
graph.add_edge(0, 1, weight=1000)
graph.add_edge(1, 2, weight=1000)
graph.add_edge(2, 3, weight=1000)
graph.add_edge(3, 0, weight=1000)
graph.add_edge(0, 3, weight=1000)
graph.add_edge(3, 2, weight=1000)
graph.add_edge(2, 1, weight=1000)
graph.add_edge(1, 0, weight=1000)

rs = np.random.RandomState()

dm_memory_length = 10
num_demands = graph.number_of_nodes() * (graph.number_of_nodes() - 1)
num_edges = graph.number_of_edges()
action_size = num_demands * num_edges
observation_shape = (dm_memory_length * num_demands,)

dm_generator_getter = lambda: dm.cyclical_sequence(
    lambda: dm.bimodal_demand(num_demands, rs), 40, 5, 0.2, rs)

### ACTUALLY TRAIN

from rlpyt.algos.pg.ppo import PPO
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context
from rlpyt.envs.gym import make as gym_make
from rlpyt.samplers.serial.sampler import SerialSampler

from ddr_mlp_agent import DdrMlpAgent


def build_and_train(env_id="ddr-v0", run_ID=0, cuda_idx=None):
    affinity = dict(cuda_idx=cuda_idx)
    sampler = SerialSampler(
        EnvCls=gym_make,
        env_kwargs=dict(id=env_id,
                        dm_generator_getter=dm_generator_getter,
                        dm_memory_length=dm_memory_length,
                        graph=graph),
        eval_env_kwargs=dict(id=env_id),
        batch_T=8,
        batch_B=4,
        max_decorrelation_steps=0,
    )
    algo = PPO()
    agent = DdrMlpAgent(
        model_kwargs={"observation_shape": observation_shape,
                      "action_size": action_size})
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=80000,
        log_interval_steps=1e3,
        affinity=affinity,
    )
    config = dict()
    name = "ppo_" + env_id
    log_dir = "ddr_mlp"
    with logger_context(log_dir, run_ID, name, config, snapshot_mode="last"):
        runner.train()


def run_experiment():
    build_and_train()


if __name__ == "__main__":
    build_and_train()
