from typing import List

import networkx as nx
import numpy as np
import tensorflow as tf
from graph_nets.demos.models import EncodeProcessDecode
from graph_nets.graphs import GraphsTuple
from stable_baselines.common.policies import ActorCriticPolicy, \
    RecurrentActorCriticPolicy
from stable_baselines.common.policies import mlp_extractor
from stable_baselines.common.policies import nature_cnn
from stable_baselines.common.tf_layers import linear, ortho_init, _ln
from stable_baselines.common.tf_util import batch_to_seq, seq_to_batch


def repeat_inner_dim(input: tf.Tensor, repeats: tf.Tensor) -> tf.Tensor:
    """
    Take a 2D tensor eg [[1,2],[3,4]] and repeat the inside n times eg
    [[1,2],[3,4],[1,2],[3,4],......]
    Args:
        input: A 2D tensor
        repeats: Number of times to repeat (a tf scalar)

    Returns:
        A tensor with the elements of the inner dimension repeated
    """
    minus_one = tf.constant(-1)
    inner_size = tf.shape(input)[1]
    reshape_tensor = tf.stack([minus_one, inner_size])

    expanded = tf.expand_dims(input, axis=0)
    repeated = tf.repeat(expanded, repeats, axis=0)
    reshaped_inner = tf.reshape(repeated, reshape_tensor)
    return reshaped_inner


def repeat_outer_dim(input: tf.Tensor, repeats: tf.Tensor) -> tf.Tensor:
    """
    Take a 1D tensor eg [1,2,3] and repeat its content n times eg
    [1,2,3,1,2,3,1,2,3.......]
    Args:
        input: A 1D tensor
        repeats: Number of times to repead (a tf scalar)

    Returns:
        A tensor with the elements repeated
    """
    expanded = tf.expand_dims(input, axis=0)
    repeated = tf.repeat(expanded, repeats, axis=0)
    reshaped_inner = tf.reshape(repeated, [-1])
    return reshaped_inner


def vf_builder(vf_arch: str, graph: nx.DiGraph, latent: tf.Tensor,
               act_fun: tf.function, shared_graphs: List[GraphsTuple] = None,
               input_graph: GraphsTuple = None,
               iterations: int = 10) -> tf.Tensor:
    """
    Builds the value function network for
    Args:
        vf_arch: arch to use as a string
        graph: the graph this is being built for
        latent: the observation input
        act_fun: activation function
        shared_graphs: the gnn output from the policy
        input_graph: GraphTuple before any processing
        iterations: number of iterations of message passing
    Returns:
        A tensor which will hold the value
    """
    num_edges = graph.number_of_edges()
    num_nodes = graph.number_of_nodes()

    if vf_arch == "shared":
        output_edges_vf = tf.reshape(shared_graphs[-1].edges,
                                     tf.constant([-1, num_edges], np.int32))
        output_nodes_vf = tf.reshape(shared_graphs[-1].nodes,
                                     tf.constant([-1, num_nodes], np.int32))
        output_globals_vf = tf.reshape(shared_graphs[-1].globals,
                                       tf.constant([-1, 1], np.int32))
        latent_vf = act_fun(
            linear(
                tf.concat([output_edges_vf, output_nodes_vf, output_globals_vf],
                          1), "vf_fc0", 128, init_scale=np.sqrt(2)))
    elif vf_arch == "graph":
        model_vf = EncodeProcessDecode(edge_output_size=1, node_output_size=1)
        output_graphs_vf = model_vf(input_graph, iterations)
        output_edges_vf = tf.reshape(output_graphs_vf[-1].edges,
                                     tf.constant([-1, num_edges], np.int32))
        output_nodes_vf = tf.reshape(output_graphs_vf[-1].nodes,
                                     tf.constant([-1, num_nodes], np.int32))
        latent_vf = act_fun(
            linear(tf.concat([output_edges_vf, output_nodes_vf], 1), "vf_fc0",
                   128, init_scale=np.sqrt(2)))
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
    node_features = tf.reshape(latent, [-1, (num_nodes - 1) * dm_memory_length],
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

    model = EncodeProcessDecode(edge_output_size=1, node_output_size=1,
                                global_output_size=1)
    output_graphs = model(input_graph, iterations)
    # NB: reshape needs num_edges as otherwise output tensor has too many
    #     unknown dims
    output_edges = tf.reshape(output_graphs[-1].edges,
                              tf.constant([-1, num_edges], np.int32))
    latent_policy_gnn = output_edges

    # build value function network
    latent_vf = vf_builder(vf_arch, network_graph, latent, act_fun,
                           output_graphs, input_graph, iterations)

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
            num_nodes - 1) * dm_memory_length])
    edge_features_slice = tf.slice(latent, [0, num_nodes * (
            num_nodes - 1) * dm_memory_length], [-1, -1])

    # reshape node features to flat batches but still vector in dim 1 per node
    node_features = tf.reshape(node_features_slice,
                               [-1, (num_nodes - 1) * dm_memory_length],
                               name="node_feat_input")
    # reshape edge features to flat batches but vector in dim 1 per edge
    edge_features = tf.reshape(edge_features_slice, [-1, 2],
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
    # We still output other for use in shard part of value function
    model = EncodeProcessDecode(edge_output_size=1, node_output_size=1,
                                global_output_size=1)
    output_graphs = model(input_graph, iterations)
    output_global = tf.reshape(output_graphs[-1].globals,
                               tf.constant([-1, 1], np.int32))
    latent_policy_gnn = output_global

    # build value function network
    latent_vf = vf_builder(vf_arch, network_graph, latent, act_fun,
                           output_graphs, input_graph, iterations)

    return latent_policy_gnn, latent_vf


def custom_lstm(input_tensor, mask_tensor, cell_state_hidden, scope, n_hidden,
                init_scale=1.0, layer_norm=False):
    """
    Creates an Long Short Term Memory (LSTM) cell for TensorFlow to be used forr DDR

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


class FeedForwardPolicyWithGnn(ActorCriticPolicy):
    """
    Modification of stable-baselines FeedForwardPolicy to support gnn feature extraction

    Policy object that implements actor critic, using a feed forward neural network.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) (deprecated, use net_arch instead) The size of the Neural network for the policy
        (if None, default to [64, 64])
    :param net_arch: (list) Specification of the actor-critic policy network architecture (see mlp_extractor
        documentation for details).
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("gnn", "gnn_iter", "cnn" or "mlp")
    :param network_graph: (nx.DiGraph) graph to use in gnn
    :param dm_memory_length: (int) length of demand matrix memory
    :param iterations: (int) Number of message passing iterations if gnn feature extraction
    :param vf_arch: (str) architecture to use for value funciton if gnn feature extraction
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                 reuse=False, layers=None,
                 net_arch=[dict(vf=[128, 128, 128], pi=[128, 128, 128])],
                 act_fun=tf.tanh, cnn_extractor=nature_cnn,
                 feature_extraction="cnn", network_graph=None,
                 dm_memory_length=None, iterations=10, vf_arch="graph",
                 **kwargs):
        super(FeedForwardPolicyWithGnn, self).__init__(sess, ob_space, ac_space,
                                                       n_env, n_steps, n_batch,
                                                       reuse=reuse, scale=(
                        feature_extraction == "cnn"))

        self._kwargs_check(feature_extraction, kwargs)

        with tf.variable_scope("model", reuse=reuse):
            if feature_extraction == "cnn":
                pi_latent = vf_latent = cnn_extractor(self.processed_obs,
                                                      **kwargs)
            elif feature_extraction == "gnn":
                pi_latent, vf_latent = gnn_extractor(
                    tf.layers.flatten(self.processed_obs), act_fun,
                    network_graph, dm_memory_length, iterations=iterations,
                    vf_arch=vf_arch)
            elif feature_extraction == "gnn_iter":
                pi_latent, vf_latent = gnn_iter_extractor(
                    tf.layers.flatten(self.processed_obs), act_fun,
                    network_graph, dm_memory_length, iterations=iterations,
                    vf_arch=vf_arch)
            else:  # Assume mlp feature extraction
                pi_latent, vf_latent = mlp_extractor(
                    tf.layers.flatten(self.processed_obs), net_arch, act_fun)

            self._value_fn = linear(vf_latent, 'vf', 1)

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent,
                                                           init_scale=0.01)

        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run(
                [self.deterministic_action, self.value_flat, self.neglogp],
                {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run(
                [self.action, self.value_flat, self.neglogp],
                {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})


class LstmPolicyWithGnn(RecurrentActorCriticPolicy):
    """
    Modification of stable-baselines LstmPolicy to support gnn feature extraction

    Policy object that implements actor critic, using LSTMs.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) The size of the Neural network before the LSTM layer  (if None, default to [64, 64])
    :param net_arch: (list) Specification of the actor-critic policy network architecture. Notation similar to the
        format described in mlp_extractor but with additional support for a 'lstm' entry in the shared network part.
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param layer_norm: (bool) Whether or not to use layer normalizing LSTMs
    :param feature_extraction: (str) The feature extraction type ("gnn", "gnn_iter", "cnn" or "mlp")
    :param network_graph: (nx.DiGraph) graph to use in gnn
    :param iterations: (int) Number of message passing iterations if gnn feature extraction
    :param vf_arch: (str) architecture to use for value funciton if gnn feature extraction
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    recurrent = True

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                 reuse=False, layers=None,
                 net_arch=[dict(vf=[128, 128, 128], pi=[128, 128, 128])],
                 act_fun=tf.tanh, cnn_extractor=nature_cnn,
                 layer_norm=False, feature_extraction="cnn",
                 network_graph=None, iterations=10, vf_arch="graph",
                 **kwargs):
        n_lstm = network_graph.number_of_nodes() * (
                network_graph.number_of_nodes() - 1)
        # state_shape = [n_lstm * 2] dim because of the cell and hidden states of the LSTM
        super(LstmPolicyWithGnn, self).__init__(sess, ob_space, ac_space, n_env,
                                                n_steps, n_batch,
                                                state_shape=(2 * n_lstm,),
                                                reuse=reuse,
                                                scale=(
                                                        feature_extraction == "cnn"))

        self._kwargs_check(feature_extraction, kwargs)

        if feature_extraction == "cnn":
            raise NotImplementedError()

        with tf.variable_scope("model", reuse=reuse):
            latent = tf.layers.flatten(self.processed_obs)

            if feature_extraction == "gnn_iter":
                # first we split the dm and embedding inputs
                num_nodes = network_graph.number_of_nodes()
                demand_matrices = tf.slice(latent, [0, 0],
                                           [-1, num_nodes * (num_nodes - 1)])
                node_embedding = tf.slice(latent,
                                          [0, num_nodes * (num_nodes - 1)],
                                          [-1, -1])

                # then we lstm the dm inputs
                input_sequence = batch_to_seq(demand_matrices, self.n_env,
                                              n_steps)
                masks = batch_to_seq(self.dones_ph, self.n_env, n_steps)
                rnn_output, self.snew = custom_lstm(input_sequence, masks,
                                                    self.states_ph, 'lstm1',
                                                    n_lstm,
                                                    layer_norm=layer_norm)
                demand_matrices = seq_to_batch(rnn_output)

                # finally we stick it all back together again
                latent = tf.concat([demand_matrices, node_embedding], axis=1)
            else:
                # start off with building lstm layer over the inputs
                input_sequence = batch_to_seq(latent, self.n_env, n_steps)
                masks = batch_to_seq(self.dones_ph, self.n_env, n_steps)
                rnn_output, self.snew = custom_lstm(input_sequence, masks,
                                                    self.states_ph, 'lstm1',
                                                    n_lstm,
                                                    layer_norm=layer_norm)
                latent = seq_to_batch(rnn_output)

            # TODO: find way to get around fact that LSTM sees same input on repeat in iter case
            # TODO: deal with variable size input (i.e. if we want to change the graph...)

            if feature_extraction == "gnn":
                pi_latent, vf_latent = gnn_extractor(
                    latent, act_fun,
                    network_graph, 1, iterations=iterations,
                    vf_arch=vf_arch)
            elif feature_extraction == "gnn_iter":
                pi_latent, vf_latent = gnn_iter_extractor(
                    latent, act_fun,
                    network_graph, 1, iterations=iterations,
                    vf_arch=vf_arch)
            else:  # Assume mlp feature extraction
                pi_latent, vf_latent = mlp_extractor(latent, net_arch, act_fun)

            self._value_fn = linear(vf_latent, 'vf', 1)
            # TODO: why not init_scale = 0.001 here like in the feedforward
            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent,
                                                           vf_latent)
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            return self.sess.run(
                [self.deterministic_action, self.value_flat, self.snew,
                 self.neglogp],
                {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})
        else:
            return self.sess.run(
                [self.action, self.value_flat, self.snew, self.neglogp],
                {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba,
                             {self.obs_ph: obs, self.states_ph: state,
                              self.dones_ph: mask})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat,
                             {self.obs_ph: obs, self.states_ph: state,
                              self.dones_ph: mask})


class MlpDdrPolicy(FeedForwardPolicyWithGnn):
    """
    Policy for data driven routing using a GCN. Idea is that inputs are a vector
    of demands to each destination given to each node. Then outputs are a value
    at each edge to be used as a "softmin" splitting ratio. This will hopefully
    give some level of generalisability.
    """

    def __init__(self, *args, **kwargs):
        super(MlpDdrPolicy, self).__init__(*args, **kwargs,
                                           feature_extraction="mlp")


class GnnDdrPolicy(FeedForwardPolicyWithGnn):
    """
    Policy for data driven routing using a GCN. Idea is that inputs are a vector
    of demands to each destination given to each node. Then outputs are a value
    at each edge to be used as a "softmin" splitting ratio. This will hopefully
    give some level of generalisability.
    """

    def __init__(self, *args, **kwargs):
        super(GnnDdrPolicy, self).__init__(*args, **kwargs,
                                           feature_extraction="gnn")


class GnnDdrIterativePolicy(FeedForwardPolicyWithGnn):
    """
    Policy for data driven routing using a GCN. Idea is that inputs are a vector
    of demands to each destination given to each node. Then outputs are a value
    at each edge to be used as a "softmin" splitting ratio. This will hopefully
    give some level of generalisability.
    """

    def __init__(self, *args, **kwargs):
        super(GnnDdrIterativePolicy, self).__init__(*args, **kwargs,
                                                    feature_extraction="gnn_iter")


class MlpLstmDdrPolicy(LstmPolicyWithGnn):
    def __init__(self, *args, **kwargs):
        super(MlpLstmDdrPolicy, self).__init__(*args, **kwargs,
                                               feature_extraction="mlp")


class GnnLstmDdrPolicy(LstmPolicyWithGnn):
    def __init__(self, *args, **kwargs):
        super(GnnLstmDdrPolicy, self).__init__(*args, **kwargs,
                                               feature_extraction="gnn")


class GnnLstmDdrIterativePolicy(LstmPolicyWithGnn):
    def __init__(self, *args, **kwargs):
        super(GnnLstmDdrIterativePolicy, self).__init__(*args, **kwargs,
                                                        feature_extraction="gnn_iter")
