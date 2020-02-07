from rlpyt.agents.pg.categorical import CategoricalPgAgent
from models.categorical_pg import CategoricalPgModel

class CartPolePgAgent(CategoricalPgAgent):
    def __init__(self, **kwargs):
        super().__init__(ModelCls=CategoricalPgModel, model_kwargs={"observation_shape":(4,), "action_size":2}, **kwargs)
