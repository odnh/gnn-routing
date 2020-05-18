from typing import List

import numpy as np
import networkx as nx
import tensorflow as tf
from graph_nets.graphs import GraphsTuple
from stable_baselines.a2c.utils import linear
from stable_baselines.common.tf_layers import ortho_init, _ln
from stable_baselines_ddr.tensor_transformations import repeat_inner_dim, repeat_outer_dim
from stable_baselines_ddr.graph_nets import DDRGraphNetwork


def vf_builder(vf_arch: str, graph: nx.DiGraph, latent: tf.Tensor,
               act_fun: tf.function, shared_graph: GraphsTuple = None,
               input_graph: GraphsTuple = None,
               iterations: int = 10) -> tf.Tensor:
    """
    Builds the value function network for
    Args:
        vf_arch: arch to use as a string
        graph: the graph this is being built for
        latent: the observation input
        act_fun: activation function
        shared_graph: the gnn output from the policy
        input_graph: GraphTuple before any processing
        iterations: number of iterations of message passing
    Returns:
        A tensor which will hold the value
    """
    num_edges = graph.number_of_edges()
    num_nodes = graph.number_of_nodes()

    if vf_arch == "shared":
        output_edges_vf = tf.reshape(shared_graph.edges,
                                     tf.constant([-1, num_edges], np.int32))
        output_nodes_vf = tf.reshape(shared_graph.nodes,
                                     tf.constant([-1, num_nodes], np.int32))
        output_globals_vf = tf.reshape(shared_graph.globals,
                                       tf.constant([-1, 1], np.int32))
        latent_vf = tf.concat(
            [output_edges_vf, output_nodes_vf, output_globals_vf], 1)
        latent_vf = act_fun(
            linear(latent_vf, "vf_fc0", 128, init_scale=np.sqrt(2)))
        latent_vf = act_fun(
            linear(latent_vf, "vf_fc1", 128, init_scale=np.sqrt(2)))
    if vf_arch == "shared_iter":
        output_edges_vf = tf.reshape(shared_graph.edges,
                                     tf.constant([-1, num_edges], np.int32))
        output_nodes_vf = tf.reshape(shared_graph.nodes,
                                     tf.constant([-1, num_nodes], np.int32))
        output_globals_vf = tf.reshape(shared_graph.globals,
                                       tf.constant([-1, 2], np.int32))
        latent_vf = tf.concat(
            [output_edges_vf, output_nodes_vf, output_globals_vf], 1)
        latent_vf = act_fun(
            linear(latent_vf, "vf_fc0", 128, init_scale=np.sqrt(2)))
        latent_vf = act_fun(
            linear(latent_vf, "vf_fc1", 128, init_scale=np.sqrt(2)))
    elif vf_arch == "graph":
        model_vf = DDRGraphNetwork(edge_output_size=8, node_output_size=8, global_output_size=8)
        output_graph_vf = model_vf(input_graph, iterations)
        output_edges_vf = tf.reshape(output_graph_vf.edges,
                                     tf.constant([-1, num_edges * 8], np.int32))
        output_nodes_vf = tf.reshape(output_graph_vf.nodes,
                                     tf.constant([-1, num_nodes * 8], np.int32))
        output_globals_vf = tf.reshape(output_graph_vf.globals,
                                       tf.constant([-1, 8], np.int32))
        latent_vf = tf.concat([output_edges_vf, output_nodes_vf, output_globals_vf], 1)
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
                  network_graph: nx.MultiDiGraph, dm_memory_length: int,
                  iterations: int = 10, vf_arch: str = "mlp"):
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
    latent = flat_observations

    sorted_edges = sorted(network_graph.edges())
    num_edges = len(sorted_edges)
    num_nodes = network_graph.number_of_nodes()
    num_batches = tf.shape(latent)[0]

    # slice the data dimension to split the edge and node features
    node_features = tf.reshape(latent, [-1, 2 * dm_memory_length],
                               name="node_feat_input")

    # initialise unused input features to all zeros
    edge_features = repeat_inner_dim(
        tf.constant(np.zeros((num_edges, 1)), np.float32), num_batches)
    global_features = repeat_inner_dim(
        tf.constant(np.zeros((1, 1)), np.float32), num_batches)

    # repeat edge information across batches and flattened for graph_nets
    sender_nodes = repeat_outer_dim(
        tf.constant(np.array([e[0] for e in sorted_edges]), np.int32),
        num_batches)
    receiver_nodes = repeat_outer_dim(
        tf.constant(np.array([e[1] for e in sorted_edges]), np.int32),
        num_batches)

    # repeat graph information across batches and flattened for graph_nets
    n_node_list = tf.reshape(
        repeat_inner_dim(tf.constant(np.array([[num_nodes]]), np.int32),
                         num_batches), [-1])
    n_edge_list = tf.reshape(
        repeat_inner_dim(tf.constant(np.array([[num_edges]]), np.int32),
                         num_batches), [-1])

    input_graph = GraphsTuple(nodes=node_features,
                              edges=edge_features,
                              globals=global_features,
                              senders=sender_nodes,
                              receivers=receiver_nodes,
                              n_node=n_node_list,
                              n_edge=n_edge_list)

    model = DDRGraphNetwork(edge_output_size=1, node_output_size=1,
                                global_output_size=1)
    output_graph = model(input_graph, iterations)
    # NB: reshape needs num_edges as otherwise output tensor has too many
    #     unknown dims
    output_edges = tf.reshape(output_graph.edges,
                              tf.constant([-1, num_edges], np.int32))
    # global output is softmin gamma
    output_globals = tf.reshape(output_graph.globals,
                                tf.constant([-1, 1], np.int32))
    latent_policy_gnn = tf.concat([output_edges, output_globals], axis=1)

    # build value function network
    latent_vf = vf_builder(vf_arch, network_graph, latent, act_fun,
                           output_graph, input_graph, iterations)

    return latent_policy_gnn, latent_vf


def gnn_iter_extractor(flat_observations: tf.Tensor, act_fun: tf.function,
                       network_graph: nx.MultiDiGraph, dm_memory_length: int,
                       iterations: int = 10, vf_arch: str = "mlp"):
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
    latent = flat_observations

    sorted_edges = sorted(network_graph.edges())
    num_edges = len(sorted_edges)
    num_nodes = network_graph.number_of_nodes()
    num_batches = tf.shape(latent)[0]

    # slice the data dimension to split the edge and node features
    node_features_slice = tf.slice(latent, [0, 0], [-1, num_nodes * (
        2) * dm_memory_length])
    edge_features_slice = tf.slice(latent, [0, num_nodes * (
        2) * dm_memory_length], [-1, -1])

    # reshape node features to flat batches but still vector in dim 1 per node
    node_features = tf.reshape(node_features_slice,
                               [-1, 2 * dm_memory_length],
                               name="node_feat_input")
    # reshape edge features to flat batches but vector in dim 1 per edge
    edge_features = tf.reshape(edge_features_slice, [-1, 3],
                               name="edge_feat_input")

    # initialise global input features to zeros (as are unused)
    global_features = repeat_inner_dim(
        tf.constant(np.zeros((1, 1)), np.float32), num_batches)

    # repeat edge information across batches and flattened for graph_nets
    sender_nodes = repeat_outer_dim(
        tf.constant(np.array([e[0] for e in sorted_edges]), np.int32),
        num_batches)
    receiver_nodes = repeat_outer_dim(
        tf.constant(np.array([e[1] for e in sorted_edges]), np.int32),
        num_batches)

    # repeat graph information across batches and flattened for graph_nets
    n_node_list = tf.reshape(
        repeat_inner_dim(tf.constant(np.array([[num_nodes]]), np.int32),
                         num_batches), [-1])
    n_edge_list = tf.reshape(
        repeat_inner_dim(tf.constant(np.array([[num_edges]]), np.int32),
                         num_batches), [-1])

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
    model = DDRGraphNetwork(edge_output_size=1, node_output_size=1,
                                global_output_size=2)
    output_graph = model(input_graph, iterations)
    output_global = tf.reshape(output_graph.globals,
                               tf.constant([-1, 2], np.int32))
    latent_policy_gnn = output_global

    # build value function network
    latent_vf = vf_builder(vf_arch, network_graph, latent, act_fun,
                           output_graph, input_graph, iterations)

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
