import torch
import dgl
from dgl.nn.pytorch.conv import GraphConv
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims


class IntermediateGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cartpole_graph = dgl.DGLGraph()
        self.cartpole_graph.add_nodes(4)
        # fully connected
        self.cartpole_graph.add_edges([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                                      [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2])
        self.model = GraphConv(1, 1)

    def forward(self, input):
        # The GraphConv takes inputs in the form of columns where batched dims
        # are subsequent dims need to present a similar API to nn.Linear,
        # nn.Softmax. Therefore transpose and unsqueeze for prep, opposite for
        # return
        input_for_gcn = input.t().unsqueeze(-1)
        ouput_from_gcn = self.model(self.cartpole_graph, input_for_gcn)
        output = ouput_from_gcn.squeeze(-1).t()
        return output


class CartPoleGcnModel(torch.nn.Module):
    """
    Graph Convolutional Model for CartPole
    NN is GCN where each input is a node, linear layer takes to ouput size,
    final softmax for use with categorical distribution.
    Value function is modeled in similar way witih single output
    """

    def __init__(self):
        super().__init__()
        self._output_size = 2
        self._obs_ndim = 1

        self.cartpole_graph = dgl.DGLGraph()
        self.cartpole_graph.add_nodes(4)
        self.cartpole_graph.add_edges([0, 1, 3], [2, 0, 2])

        self.pi = torch.nn.Sequential(
            IntermediateGCN(),
            torch.nn.Linear(4, 2),
            torch.nn.Softmax(dim=-1))

        self.v = torch.nn.Sequential(
            IntermediateGCN(),
            torch.nn.Linear(4, 1),
            torch.nn.ReLU())

    def forward(self, observation, prev_action, prev_reward):
        """
        Compute the model on the input, assuming input shape [B,input_size].
        NB: as graph does not change, we have 'IntermediateGCN' to hold and pass
            in the graph during evaulation (and manage API differences)
        """
        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)
        obs_flat = observation.view(T * B, -1)

        pi = self.pi(obs_flat).squeeze(-1)
        v = self.v(obs_flat).squeeze(-1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)

        return pi, v
