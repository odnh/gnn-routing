from gym.envs.registration import register

register(
        id='ddr-v0',
        entry_point='gym_gnn.envs:DDREnv',
        )
