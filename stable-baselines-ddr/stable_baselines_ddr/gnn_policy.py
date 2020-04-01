import warnings
from typing import List

import networkx as nx
import numpy as np
import tensorflow as tf
from graph_nets.demos.models import EncodeProcessDecode
from graph_nets.graphs import GraphsTuple
from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.common.policies import mlp_extractor
from stable_baselines.common.policies import nature_cnn
from stable_baselines.common.tf_layers import linear


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


def gnn_extractor(flat_observations: tf.Tensor, net_arch: List,
                  act_fun: tf.function, network_graph: nx.MultiDiGraph, dm_memory_length: int):
    """
    TODO: rewrite the whole docstring
    Constructs an MLP that receives observations as an input and outputs a latent representation for the policy and
    a value network. The ``net_arch`` parameter allows to specify the amount and size of the hidden layers and how many
    of them are shared between the policy network and the value network. It is assumed to be a list with the following
    structure:

    1. An arbitrary length (zero allowed) number of integers each specifying the number of units in a shared layer.
       If the number of ints is zero, there will be no shared layers.
    2. An optional dict, to specify the following non-shared layers for the value network and the policy network.
       It is formatted like ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``.
       If it is missing any of the keys (pi or vf), no non-shared layers (empty list) is assumed.

    For example to construct a network with one shared layer of size 55 followed by two non-shared layers for the value
    network of size 255 and a single non-shared layer of size 128 for the policy network, the following layers_spec
    would be used: ``[55, dict(vf=[255, 255], pi=[128])]``. A simple shared network topology with two layers of size 128
    would be specified as [128, 128].

    :param flat_observations: (tf.Tensor) The observations to base policy and value function on.
    :param net_arch: ([int or dict]) The specification of the policy and value networks.
        See above for details on its formatting.
    :param act_fun: (tf function) The activation function to use for the networks.
    :param network_graph: (nx.DiGraph) The graph to base the model on.
    :return: (tf.Tensor, tf.Tensor) latent_policy, latent_value of the specified network.
        If all layers are shared, then ``latent_policy == latent_value``
    """
    # TODO: use separate network for value function (maybe mlpness)
    latent = flat_observations

    sorted_edges = sorted(network_graph.edges())
    num_edges = len(sorted_edges)
    num_nodes = network_graph.number_of_nodes()
    num_batches = tf.shape(latent)[0]

    node_features = tf.reshape(latent, [-1, (num_nodes - 1) * dm_memory_length],
                               name="node_feat_input")
    edge_features = repeat_inner_dim(
        tf.constant(np.zeros((num_edges, 1)), np.float32), num_batches)
    global_features = repeat_inner_dim(
        tf.constant(np.zeros((1, 1)), np.float32), num_batches)
    sender_nodes = repeat_outer_dim(
        tf.constant(np.array([e[0] for e in sorted_edges]), np.int32),
        num_batches)
    receiver_nodes = repeat_outer_dim(
        tf.constant(np.array([e[1] for e in sorted_edges]), np.int32),
        num_batches)
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

    model = EncodeProcessDecode(edge_output_size=1)
    output_graphs = model(input_graph, 1)
    # NB: reshape needs num_edges as otherwise output tensor has too many
    #     unknown dims
    output_edges = tf.reshape(output_graphs[0].edges,
                              tf.constant([-1, num_edges], np.int32))
    latent_policy_gnn = output_edges

    return latent_policy_gnn, latent_policy_gnn


class FeedForwardPolicyWithGnn(ActorCriticPolicy):
    """
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
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                 reuse=False, layers=None, net_arch=None,
                 act_fun=tf.tanh, cnn_extractor=nature_cnn,
                 feature_extraction="cnn", network_graph=None, dm_memory_length=None, **kwargs):
        super(FeedForwardPolicyWithGnn, self).__init__(sess, ob_space, ac_space,
                                                       n_env,
                                                       n_steps, n_batch,
                                                       reuse=reuse,
                                                       scale=(
                                                               feature_extraction == "cnn"))

        self._kwargs_check(feature_extraction, kwargs)

        if layers is not None:
            warnings.warn(
                "Usage of the `layers` parameter is deprecated! Use net_arch instead "
                "(it has a different semantics though).", DeprecationWarning)
            if net_arch is not None:
                warnings.warn(
                    "The new `net_arch` parameter overrides the deprecated `layers` parameter!",
                    DeprecationWarning)

        if net_arch is None:
            if layers is None:
                layers = [64, 64]
            net_arch = [dict(vf=layers, pi=layers)]

        with tf.variable_scope("model", reuse=reuse):
            if feature_extraction == "cnn":
                pi_latent = vf_latent = cnn_extractor(self.processed_obs,
                                                      **kwargs)
            elif feature_extraction == "gnn":
                pi_latent, vf_latent = gnn_extractor(
                    tf.layers.flatten(self.processed_obs), net_arch, act_fun,
                    network_graph, dm_memory_length)
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


class GnnDdrPolicy(FeedForwardPolicyWithGnn):
    """
    Policy for data driven routing using a GCN. Idea is that inputs are a vector
    of demands to each destination given to each node. Then outputs are a value
    at each edge to be used as a "softmin" splitting ratio. This will hopefully
    give some level of generalisability.
    """

    def __init__(self, *args, **kwargs):
        super(GnnDdrPolicy, self).__init__(*args, **kwargs,
                                           # net_arch=[dict(pi=[128, 128, 128],
                                           #                vf=[128, 128, 128])],
                                           net_arch=[64, 64, 8],
                                           feature_extraction="gnn")
