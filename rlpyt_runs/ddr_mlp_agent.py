from rlpyt.agents.pg.gaussian import GaussianPgAgent
from ddr_mlp_model import DdrMlpModel

class DdrMlpAgent(GaussianPgAgent):
    def __init__(self, model_kwargs={}, **kwargs):
        super().__init__(ModelCls=DdrMlpModel, model_kwargs=model_kwargs, **kwargs)