"""
Train DDR for destination-based or softmin routing using PPO
"""
import gym_gnn.envs.demand_matrices as dm
import networkx as nx
import numpy as np
from rlpyt.algos.pg.ppo import PPO
from rlpyt.envs.gym import make as gym_make
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.utils.logging.context import logger_context

from rlpyt_runs.agents.mlp import DdrMlpAgent
from rlpyt_runs.graphs.generator import basic


def run_experiment(env: str = "softmin", graph: nx.DiGraph = basic(),
                   run_ID: int = 0, cuda_idx: int = None):
    rs = np.random.RandomState()

    dm_memory_length = 10
    num_demands = graph.number_of_nodes() * (graph.number_of_nodes() - 1)
    num_edges = graph.number_of_edges()
    out_edge_count = [i[1] for i in graph.out_degree()]
    if env == "softmin":
        action_size = num_edges
    else:
        action_size = sum(out_edge_count) * (graph.number_of_nodes() - 1)
    observation_size = dm_memory_length * num_demands

    dm_generator_getter = lambda: dm.cyclical_sequence(
        lambda: dm.bimodal_demand(num_demands, rs), 40, 5, 0.0, rs)

    env_id = "ddr-{}-v0".format(env)

    affinity = dict(cuda_idx=cuda_idx)
    sampler = SerialSampler(
        EnvCls=gym_make,
        env_kwargs=dict(id=env_id,
                        dm_generator_getter=dm_generator_getter,
                        dm_memory_length=dm_memory_length,
                        graph=graph),
        eval_env_kwargs=dict(id=env_id),
        batch_T=6,
        batch_B=2,
        max_decorrelation_steps=4,
    )
    algo = PPO()
    agent = DdrMlpAgent(
        model_kwargs={"observation_size": observation_size,
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


if __name__ == "__main__":
    run_experiment()
