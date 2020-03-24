from rlpyt.agents.pg.gaussian import GaussianPgAgent
from rlpyt_runs.models.gnn import DdrGnnModel

class DdrGnnAgent(GaussianPgAgent):
    def __init__(self, model_kwargs={}, **kwargs):
        super().__init__(
            ModelCls=DdrGnnModel, model_kwargs=model_kwargs, **kwargs)