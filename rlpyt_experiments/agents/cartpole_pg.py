from rlpyt.agents.pg.categorical import CategoricalPgAgent
from models.cartpole_pg import CartPolePgModel

class CartPolePgAgent(CategoricalPgAgent):
    def __init__(self, **kwargs):
        super().__init__(ModelCls=CartPolePgModel, model_kwargs={"observation_shape":(4,), "action_size":2}, **kwargs)
