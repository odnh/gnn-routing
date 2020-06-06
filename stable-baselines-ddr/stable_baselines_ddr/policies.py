import tensorflow as tf
from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.common.policies import mlp_extractor
from stable_baselines.common.policies import nature_cnn
from stable_baselines.common.tf_layers import linear
from stable_baselines_ddr.feature_extraction import gnn_iter_extractor, \
    gnn_extractor


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
                 feature_extraction="gnn",
                 layer_size=64,
                 layer_count=2,
                 network_graphs=None,
                 dm_memory_length=None, iterations=10, vf_arch="mlp",
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
                    network_graphs, dm_memory_length, layer_size=layer_size, layer_count=layer_count, iterations=iterations,
                    vf_arch=vf_arch)
            elif feature_extraction == "gnn_iter":
                pi_latent, vf_latent = gnn_iter_extractor(
                    tf.layers.flatten(self.processed_obs), act_fun,
                    network_graphs, dm_memory_length, layer_size=layer_size, layer_count=layer_count, iterations=iterations,
                    vf_arch=vf_arch)
            else:  # Assume mlp feature extraction
                pi_latent, vf_latent = mlp_extractor(
                    tf.layers.flatten(self.processed_obs), net_arch, act_fun)
                # Need this here as removed from proba_distribution
                # ok to choose first as can only run mlp one one graph anyway
                pi_latent = linear(pi_latent, 'pi',
                                   network_graphs[0].number_of_edges() + 1,
                                   init_scale=0.01,
                                   init_bias=0.0)

            self._value_fn = linear(vf_latent, 'vf', 1)

            # self._proba_distribution, self._policy, self.q_value = \
            #     self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent,
            #                                                init_scale=0.01)
            self._proba_distribution, self._policy, self.q_value = \
                self.proba_distribution_no_pi_linear(pi_latent, vf_latent,
                                                     init_scale=0.01)

        self._setup_init()

    def proba_distribution_no_pi_linear(self, pi_latent_vector,
                                        vf_latent_vector, init_scale=0.01,
                                        init_bias=0.0):
        """
        Remove extra linear for debugging purposes
        """
        # mean = linear(pi_latent_vector, 'pi', self.size, init_scale=init_scale, init_bias=init_bias)
        mean = pi_latent_vector
        logstd = tf.get_variable(name='pi/logstd', shape=[1, self.pdtype.size],
                                 initializer=tf.zeros_initializer())
        pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
        q_values = linear(vf_latent_vector, 'q', self.pdtype.size,
                          init_scale=init_scale, init_bias=init_bias)
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
