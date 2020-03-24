from rlpyt.agents.pg.gaussian import GaussianPgAgent
from rlpyt_runs.models.mlp import DdrMlpModel

class DdrMlpAgent(GaussianPgAgent):
    def __init__(self, model_kwargs={}, **kwargs):
        super().__init__(
            ModelCls=DdrMlpModel, model_kwargs=model_kwargs, **kwargs)