from typing import List

import networkx as nx
import numpy as np
import tensorflow as tf
from graph_nets.graphs import GraphsTuple
from stable_baselines.a2c.utils import linear
from stable_baselines.common.tf_layers import ortho_init, _ln
from stable_baselines_ddr.graph_nets import DDRGraphNetwork


def vf_builder(vf_arch: str, latent: tf.Tensor,
               act_fun: tf.function, shared_graph: GraphsTuple = None,
               input_graph: GraphsTuple = None, layer_size: int = 64,
               layer_count: int = 3,
               iterations: int = 10) -> tf.Tensor:
    """
    Builds the value function network for
    Args:
        vf_arch: arch to use as a string
        latent: the observation input
        act_fun: activation function
        shared_graph: the gnn output from the policy
        input_graph: GraphTuple before any processing
        iterations: number of iterations of message passing
    Returns:
        A tensor which will hold the value
    """
    if vf_arch == "shared":
        output_globals_vf = tf.reshape(shared_graph.globals, [-1, layer_size])
        latent_vf = output_globals_vf
        latent_vf = act_fun(
            linear(latent_vf, "vf_fc0", 128, init_scale=np.sqrt(2)))
        latent_vf = act_fun(
            linear(latent_vf, "vf_fc1", 128, init_scale=np.sqrt(2)))
    elif vf_arch == "graph":
        model_vf = DDRGraphNetwork(layer_size=layer_size)
        output_graph_vf = model_vf(input_graph, iterations)
        output_globals_vf = tf.reshape(output_graph_vf.globals,
                                       [-1, layer_size])
        latent_vf = output_globals_vf
    elif vf_arch == "mlp":
        latent_vf = latent
        latent_vf = act_fun(
            linear(latent_vf, "vf_fc0", 128, init_scale=np.sqrt(2)))
        latent_vf = act_fun(
            linear(latent_vf, "vf_fc1", 128, init_scale=np.sqrt(2)))
        latent_vf = act_fun(
            linear(latent_vf, "vf_fc2", 128, init_scale=np.sqrt(2)))
    else:
        raise Exception("No such vf network")

    return latent_vf


def gnn_extractor(flat_observations: tf.Tensor, act_fun: tf.function,
                  network_graphs: List[nx.DiGraph], dm_memory_length: int,
                  iterations: int = 10, layer_size: int = 128,
                  layer_count: int = 3,
                  vf_arch: str = "mlp"):
    """
    Constructs a graph network from the graph passed in. Then inputs are
    traffic demands, placed on nodes as feature vectors. The output policy
    tensor is built from the edge outputs (in line with the softmin routing
    approach). The value function can be switched between mlp and graph net
    using the net_arch argument.

    :return: (tf.Tensor, tf.Tensor) latent_policy, latent_value of the
    specified network. If all layers are shared, then ``latent_policy ==
    latent_value``
    """
    # get graph info
    sorted_edges_list = [sorted(network_graph.edges()) for network_graph in
                         network_graphs]
    num_edges_list = [len(l) for l in sorted_edges_list]
    num_nodes_list = [network_graph.number_of_nodes() for network_graph in
                      network_graphs]
    max_edges_len = max(num_edges_list)
    sorted_edges_list = [edges + [(0, 0)] * (max_edges_len - len(edges)) for
                         edges in sorted_edges_list]
    sorted_edges = tf.constant(sorted_edges_list)
    num_edges = tf.constant(num_edges_list)
    num_nodes = tf.constant(num_nodes_list)

    # start manipulating input data
    latent = flat_observations
    num_batches = tf.shape(latent)[0]

    # prepare helper data for graphs per entry in batch
    graph_idxs = tf.cast(latent[:, 0], np.int32)
    num_nodes_per_batch = tf.map_fn(lambda i: num_nodes[i], graph_idxs)
    num_edges_per_batch = tf.map_fn(lambda i: num_edges[i], graph_idxs)
    observation_sizes = tf.multiply(num_nodes_per_batch, dm_memory_length * 2)
    full_observations = latent[:, 1:]
    trimmed_observations = tf.RaggedTensor.from_tensor(full_observations,
                                                       lengths=observation_sizes)

    # reshape data into correct sizes for gnn input
    node_features = tf.reshape(trimmed_observations.flat_values,
                               [-1, 2 * dm_memory_length],
                               name="node_feat_input")
    node_features = tf.pad(node_features,
                           [[0, 0], [0, layer_size - (2 * dm_memory_length)]])

    # initialise unused input features to all zeros
    edge_features = tf.zeros((tf.reduce_sum(num_edges_per_batch), layer_size),
                             np.float32)
    global_features = tf.zeros((num_batches, layer_size), np.float32)

    # repeat edge information across batches and flattened for graph_nets
    sender_nodes = tf.map_fn(lambda i: sorted_edges[i][:, 0], graph_idxs)
    sender_nodes = tf.RaggedTensor.from_tensor(sender_nodes,
                                               lengths=num_edges_per_batch)
    sender_nodes = sender_nodes.flat_values
    receiver_nodes = tf.map_fn(lambda i: sorted_edges[i][:, 1], graph_idxs)
    receiver_nodes = tf.RaggedTensor.from_tensor(receiver_nodes,
                                                 lengths=num_edges_per_batch)
    receiver_nodes = receiver_nodes.flat_values

    # repeat graph information across batches and flattened for graph_nets
    n_node_list = num_nodes_per_batch
    n_edge_list = num_edges_per_batch

    input_graph = GraphsTuple(nodes=node_features,
                              edges=edge_features,
                              globals=global_features,
                              senders=sender_nodes,
                              receivers=receiver_nodes,
                              n_node=n_node_list,
                              n_edge=n_edge_list)

    model = DDRGraphNetwork(layer_size=layer_size, layer_count=layer_count)
    output_graph = model(input_graph, iterations)

    # NB: reshape needs num_edges as otherwise output tensor has too many
    #     unknown dims
    # first split per graph
    output_edges = tf.RaggedTensor.from_row_lengths(output_graph.edges,
                                                    num_edges_per_batch)
    # make a normal tensor so we can slice out edge values
    output_edges = output_edges.to_tensor()
    # then extract from each split the values we want and squeeze away last axis
    output_edges = tf.squeeze(output_edges[:, :, 0::layer_size], axis=2)
    # finally pad to correct size for output
    output_edges = tf.pad(output_edges, [[0, 0], [0, max_edges_len -
                                                  tf.shape(output_edges)[1]]])

    # global output is softmin gamma
    output_globals = tf.reshape(output_graph.globals, (-1, layer_size))
    output_globals = output_globals[:, 0]
    output_globals = tf.reshape(output_globals, (-1, 1))

    latent_policy_gnn = tf.concat([output_edges, output_globals], axis=1)
    # build value function network
    latent_vf = vf_builder(vf_arch, flat_observations, act_fun,
                           output_graph, input_graph, layer_size, layer_count,
                           iterations)

    return latent_policy_gnn, latent_vf


def gnn_iter_extractor(flat_observations: tf.Tensor, act_fun: tf.function,
                       network_graphs: List[nx.DiGraph], dm_memory_length: int,
                       iterations: int = 10, layer_size: int = 64,
                       layer_count: int = 3,
                       vf_arch: str = "mlp"):
    """
    Constructs a graph network from the graph passed in. Then inputs are
    traffic demands, placed on nodes as feature vectors. The inputs also
    include flags as to whether and edge has been set and which one should be
    set this iteration which are placed on the edges. The output policy
    tensor is built from the edge outputs (in line with the softmin routing
    approach). The value function can be switched between mlp and graph net
    using the net_arch argument.

    :return: (tf.Tensor, tf.Tensor) latent_policy, latent_value of the
    specified network. If all layers are shared, then ``latent_policy ==
    latent_value``
    """
    # get graph info
    sorted_edges_list = [sorted(network_graph.edges()) for network_graph in
                         network_graphs]
    num_edges_list = [len(l) for l in sorted_edges_list]
    num_nodes_list = [network_graph.number_of_nodes() for network_graph in
                      network_graphs]
    max_edges_len = max(num_edges_list)
    sorted_edges_list = [edges + [(0, 0)] * (max_edges_len - len(edges)) for
                         edges in sorted_edges_list]
    sorted_edges = tf.constant(sorted_edges_list)
    num_edges = tf.constant(num_edges_list)
    num_nodes = tf.constant(num_nodes_list)

    # start manipulating the input
    latent = flat_observations
    num_batches = tf.shape(latent)[0]

    # prepare helper data for graphs per entry in batch
    graph_idxs = tf.cast(latent[:, 0], np.int32)
    num_nodes_per_batch = tf.map_fn(lambda i: num_nodes[i], graph_idxs)
    num_edges_per_batch = tf.map_fn(lambda i: num_edges[i], graph_idxs)
    observation_sizes = tf.multiply(num_nodes_per_batch,
                                    dm_memory_length * 2) + tf.multiply(
        num_edges_per_batch, 2)
    full_observations = latent[:, 1:]
    trimmed_observations = tf.RaggedTensor.from_tensor(full_observations,
                                                       lengths=observation_sizes)

    # slice apart the node and edge features in each seciton of batch
    node_observation_sizes = tf.multiply(num_nodes_per_batch,
                                         dm_memory_length * 2)
    edge_observation_sizes = tf.multiply(num_edges_per_batch, 2)
    interleaved_lengths = tf.reshape(
        tf.stack([node_observation_sizes, edge_observation_sizes], axis=1),
        [-1])
    flattened_observations = trimmed_observations.flat_values
    interleaved_observations = tf.RaggedTensor.from_row_lengths(
        flattened_observations, interleaved_lengths)
    node_features_slice = interleaved_observations[::2].flat_values
    edge_features_slice = interleaved_observations[1::2].flat_values

    # reshape and pad for input to gnn
    node_features = tf.reshape(node_features_slice, [-1, 2 * dm_memory_length])
    node_features = tf.pad(node_features,
                           [[0, 0], [0, layer_size - (2 * dm_memory_length)]])

    edge_features = tf.reshape(edge_features_slice, [-1, 2])
    edge_features = tf.pad(edge_features, [[0, 0], [0, layer_size - 2]])

    # initialise global input features to zeros (as are unused)
    global_features = tf.zeros((num_batches, layer_size), np.float32)

    # repeat edge information across batches and flattened for graph_nets
    sender_nodes = tf.map_fn(lambda i: sorted_edges[i][:, 0], graph_idxs)
    sender_nodes = tf.RaggedTensor.from_tensor(sender_nodes,
                                               lengths=num_edges_per_batch)
    sender_nodes = sender_nodes.flat_values
    receiver_nodes = tf.map_fn(lambda i: sorted_edges[i][:, 1], graph_idxs)
    receiver_nodes = tf.RaggedTensor.from_tensor(receiver_nodes,
                                                 lengths=num_edges_per_batch)
    receiver_nodes = receiver_nodes.flat_values

    # repeat graph information across batches and flattened for graph_nets
    n_node_list = num_nodes_per_batch
    n_edge_list = num_edges_per_batch

    input_graph = GraphsTuple(nodes=node_features,
                              edges=edge_features,
                              globals=global_features,
                              senders=sender_nodes,
                              receivers=receiver_nodes,
                              n_node=n_node_list,
                              n_edge=n_edge_list)

    # Our only output is a single global which is the value to set the edge
    # We still output other for use in shared part of value function
    # The global output is: [edge_value, gamma_value]
    model = DDRGraphNetwork(layer_size=layer_size, layer_count=layer_count)
    output_graph = model(input_graph, iterations)
    output_globals = tf.reshape(output_graph.globals,
                                tf.constant([-1, layer_size], np.int32))
    output_globals = output_globals[:, 0:2]
    latent_policy_gnn = output_globals

    # build value function network
    latent_vf = vf_builder(vf_arch, flat_observations, act_fun,
                           output_graph, input_graph, layer_size, layer_count,
                           iterations)

    return latent_policy_gnn, latent_vf


def custom_lstm(input_tensor, mask_tensor, cell_state_hidden, scope, n_hidden,
                init_scale=1.0, layer_norm=False):
    """
    Creates an Long Short Term Memory (LSTM) cell for TensorFlow to be used for DDR

    :param input_tensor: (TensorFlow Tensor) The input tensor for the LSTM cell
    :param mask_tensor: (TensorFlow Tensor) The mask tensor for the LSTM cell
    :param cell_state_hidden: (TensorFlow Tensor) The state tensor for the LSTM cell
    :param scope: (str) The TensorFlow variable scope
    :param n_hidden: (int) The number of hidden neurons
    :param init_scale: (int) The initialization scale
    :param layer_norm: (bool) Whether to apply Layer Normalization or not
    :return: (TensorFlow Tensor) LSTM cell
    """
    _, n_input = [v.value for v in input_tensor[0].get_shape()]
    with tf.variable_scope(scope):
        weight_x = tf.get_variable("wx", [n_input, n_hidden * 4],
                                   initializer=ortho_init(init_scale))
        weight_h = tf.get_variable("wh", [n_hidden, n_hidden * 4],
                                   initializer=ortho_init(init_scale))
        bias = tf.get_variable("b", [n_hidden * 4],
                               initializer=tf.constant_initializer(0.0))

        if layer_norm:
            # Gain and bias of layer norm
            gain_x = tf.get_variable("gx", [n_hidden * 4],
                                     initializer=tf.constant_initializer(1.0))
            bias_x = tf.get_variable("bx", [n_hidden * 4],
                                     initializer=tf.constant_initializer(0.0))

            gain_h = tf.get_variable("gh", [n_hidden * 4],
                                     initializer=tf.constant_initializer(1.0))
            bias_h = tf.get_variable("bh", [n_hidden * 4],
                                     initializer=tf.constant_initializer(0.0))

            gain_c = tf.get_variable("gc", [n_hidden],
                                     initializer=tf.constant_initializer(1.0))
            bias_c = tf.get_variable("bc", [n_hidden],
                                     initializer=tf.constant_initializer(0.0))

    cell_state, hidden = tf.split(axis=1, num_or_size_splits=2,
                                  value=cell_state_hidden)
    for idx, (_input, mask) in enumerate(zip(input_tensor, mask_tensor)):
        cell_state = cell_state * (1 - mask)
        hidden = hidden * (1 - mask)
        if layer_norm:
            gates = _ln(tf.matmul(_input, weight_x), gain_x, bias_x) \
                    + _ln(tf.matmul(hidden, weight_h), gain_h, bias_h) + bias
        else:
            gates = tf.matmul(_input, weight_x) + tf.matmul(hidden,
                                                            weight_h) + bias
        in_gate, forget_gate, out_gate, cell_candidate = tf.split(axis=1,
                                                                  num_or_size_splits=4,
                                                                  value=gates)
        in_gate = tf.nn.sigmoid(in_gate)
        forget_gate = tf.nn.sigmoid(forget_gate)
        out_gate = tf.nn.sigmoid(out_gate)
        cell_candidate = tf.tanh(cell_candidate)
        cell_state = forget_gate * cell_state + in_gate * cell_candidate
        if layer_norm:
            hidden = out_gate * tf.tanh(_ln(cell_state, gain_c, bias_c))
        else:
            hidden = out_gate * tf.tanh(cell_state)
        input_tensor[idx] = hidden
    cell_state_hidden = tf.concat(axis=1, values=[cell_state, hidden])
    return input_tensor, cell_state_hidden
