from rlpyt.agents.pg.categorical import CategoricalPgAgent
from models.ddr_mlp import DdrMlpModel

class DdrMlpAgent(CategoricalPgAgent):
    def __init__(self, **kwargs):
        super().__init__(ModelCls=DdrMlpModel, **kwargs)