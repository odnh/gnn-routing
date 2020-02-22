import numpy as np
import torch

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.mlp import MlpModel

class DdrMlpModel(torch.nn.Module):
    """
    Model for DDR using MLP with softmax output for categorical
    TODO: actually setup input, output, and structure correctly
    """

    def __init__(
            self,
            observation_shape,
            action_size,
            hidden_sizes=[64, 64]):
        """Instantiate neural net module according to inputs."""
        super().__init__()
        self._obs_ndim = len(observation_shape)
        input_size = int(np.prod(observation_shape))

        # wrap in softmax for the categorical
        self.pi = torch.nn.Sequential(
                MlpModel(
                    input_size=input_size,
                    hidden_sizes=hidden_sizes,
                    output_size=action_size,
                    nonlinearity=torch.nn.ReLU),
                torch.nn.Softmax(dim=-1))

        self.v = MlpModel(
                input_size=input_size,
                hidden_sizes=hidden_sizes,
                output_size=1,
                nonlinearity=torch.nn.ReLU)

    def forward(self, observation, prev_action, prev_reward):
        """
        Compute action probabilities and value estimate from input state.
        Infers leading dimensions of input: can be [T,B], [B], or []; provides
        returns with same leading dims. Used in both sampler and in algorithm (both
        via the agent).
        """

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)
        obs_flat = observation.view(T * B, -1)

        pi = self.pi(obs_flat).squeeze(-1)
        v = self.v(obs_flat).squeeze(-1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)

        return pi, v