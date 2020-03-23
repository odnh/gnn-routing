from rlpyt.agents.pg.gaussian import GaussianPgAgent
from ddr_mlp_model import DdrMlpModel, DdrMlpDestModel

class DdrMlpAgent(GaussianPgAgent):
    def __init__(self, model_kwargs={}, **kwargs):
        super().__init__(
            ModelCls=DdrMlpModel, model_kwargs=model_kwargs, **kwargs)

class DdrMlpDestAgent(GaussianPgAgent):
    def __init__(self, model_kwargs={}, **kwargs):
        super().__init__(
            ModelCls=DdrMlpDestModel, model_kwargs=model_kwargs, **kwargs)
