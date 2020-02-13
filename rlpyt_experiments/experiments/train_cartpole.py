"""
Train CartPole using PPO and minibatch
"""
from rlpyt.algos.pg.ppo import PPO
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context
from rlpyt.envs.gym import make as gym_make
from rlpyt.samplers.serial.sampler import SerialSampler

from agents.cartpole_gcn import CartPoleGcnAgent
from agents.cartpole_pg import CartPolePgAgent

def build_and_train(env_id="CartPole-v0", run_ID=0, cuda_idx=None):
    affinity = dict(cuda_idx=cuda_idx)
    sampler = SerialSampler(
        EnvCls=gym_make,
        env_kwargs=dict(id=env_id),
        eval_env_kwargs=dict(id=env_id),
        batch_T=8,
        batch_B=4,
        max_decorrelation_steps=2,
    )
    algo = PPO()
    agent = CartPoleGcnAgent()#CartPolePgAgent()
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
    log_dir = "cartpole_pg"
    with logger_context(log_dir, run_ID, name, config, snapshot_mode="last"):
        runner.train()

def run_experiment():
    build_and_train()

if __name__ == "__main__":
    build_and_train()
