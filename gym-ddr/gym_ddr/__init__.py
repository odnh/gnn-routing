from gym.envs.registration import register

register(
        id='ddr-v0',
        entry_point='gym_gnn.envs:DDREnv',
        )
register(
        id='ddr-dest-v0',
        entry_point='gym_gnn.envs:DDREnvDest',
)
register(
        id='ddr-softmin-v0',
        entry_point='gym_gnn.envs:DDREnvSoftmin',
)
