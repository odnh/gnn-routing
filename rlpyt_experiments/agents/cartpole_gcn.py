from rlpyt.agents.pg.categorical import CategoricalPgAgent
from models.cartpole_gcn import CartPoleGcnModel

class CartPoleGcnAgent(CategoricalPgAgent):
    def __init__(self, **kwargs):
        super().__init__(ModelCls=CartPoleGcnModel, **kwargs)
