import tensorflow as tf
from stable_baselines.common.policies import ActorCriticPolicy, \
    RecurrentActorCriticPolicy
from stable_baselines.common.policies import mlp_extractor
from stable_baselines.common.policies import nature_cnn
from stable_baselines.common.tf_layers import linear, ortho_init, _ln
from stable_baselines.common.tf_util import batch_to_seq, seq_to_batch
from stable_baselines_ddr.feature_extraction import gnn_iter_extractor, gnn_extractor


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
            # TODO: get this working (whole change of output space size thing)
                # self.proba_distribution_no_pi_linear(pi_latent, vf_latent,
                #                                      init_scale=0.01)

        self._setup_init()

    def proba_distribution_no_pi_linear(self, pi_latent_vector, vf_latent_vector, init_scale=0.01, init_bias=0.0):
        """
        Remove extra linear for debugging purposes
        """
        # mean = linear(pi_latent_vector, 'pi', self.size, init_scale=init_scale, init_bias=init_bias)
        mean = pi_latent_vector
        logstd = tf.get_variable(name='pi/logstd', shape=[1, self.pdtype.size], initializer=tf.zeros_initializer())
        pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
        q_values = linear(vf_latent_vector, 'q', self.pdtype.size, init_scale=init_scale, init_bias=init_bias)
        return self.pdtype.proba_distribution_from_flat(pdparam), mean, q_values

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
