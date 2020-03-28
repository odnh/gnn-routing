from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C


class GcnDdrPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(GcnDdrPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[128, 128, 128], vf=[128, 128, 128])],
                                           feature_extraction="mlp")
