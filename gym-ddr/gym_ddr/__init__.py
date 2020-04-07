from gym.envs.registration import register

register(
        id='ddr-v0',
        entry_point='gym_ddr.envs:DDREnv',
        )
register(
        id='ddr-dest-v0',
        entry_point='gym_ddr.envs:DDREnvDest',
)
register(
        id='ddr-softmin-v0',
        entry_point='gym_ddr.envs:DDREnvSoftmin',
)
register(
        id='ddr-iterative-v0',
        entry_point='gym_ddr.envs:DDREnvIterative',
)
