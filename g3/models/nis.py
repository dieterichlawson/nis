import google3

import functools
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from google3.experimental.users.gjt.his.models import base

class NIS(object):

  def __init__(self,
               K,
               data_dim,
               energy_hidden_sizes,
               proposal=None,
               data_mean=None,
               reparameterize_proposal_samples=True,
               dtype=tf.float32,
               name="nis"):
    """Creates a NIS model.

    Args:
      K: The number of proposal samples to take.
      data_dim: The dimension of the data.
      energy_hidden_sizes: The sizes of the hidden layers for the MLP that parameterizes the
        energy function.
      proposal: A distribution over the data space of this model. Must support sample()
        and log_prob() although log_prob only needs to return a lower bound on the true
        log probability. If not supplied, then defaults to Gaussian.
    """
    self.data_dim = data_dim
    self.reparameterize_proposal_samples = reparameterize_proposal_samples
    if data_mean is not None:
      self.data_mean = data_mean
    else:
      self.data_mean = tf.zeros((), dtype=dtype)
    self.K = K
    self.energy_fn = functools.partial(
          base.mlp,
          layer_sizes=energy_hidden_sizes + [1],
          final_activation=None,
          name="%s/energy_fn_mlp" % name)
    if proposal is None:
      self.proposal = tfd.MultivariateNormalDiag(loc=tf.zeros([data_dim], dtype=dtype),
                                                 scale_diag=tf.ones([data_dim], dtype=dtype))
    else:
      self.proposal = proposal

  def log_prob(self, data, num_samples=1):
    batch_shape = tf.shape(data)[0:-1]
    reshaped_data = tf.reshape(data, [tf.math.reduce_prod(batch_shape), self.data_dim])
    log_prob = self._log_prob(reshaped_data, num_samples=num_samples)
    log_prob = tf.reshape(log_prob, batch_shape)
    return log_prob

  def _log_prob(self, data, num_samples=1):
    batch_size = tf.shape(data)[0]
    # Sample from the proposal and compute the weighs of the "unseen" samples.
    # We share these across the batch dimension.
    # [num_samples, K, data_size]
    proposal_samples = self.proposal.sample([num_samples, self.K])
    if not self.reparameterize_proposal_samples:
      proposal_samples = tf.stop_gradient(proposal_samples)

    # [num_samples, K]
    log_energy_proposal = tf.reshape(self.energy_fn(proposal_samples - self.data_mean),
            [num_samples, self.K])
    tf.summary.histogram("log_energy_proposal", log_energy_proposal)
    tf.summary.scalar("min_log_energy_proposal", tf.reduce_min(log_energy_proposal))
    tf.summary.scalar("max_log_energy_proposal", tf.reduce_max(log_energy_proposal))
    # [num_samples]
    proposal_lse = tf.reduce_logsumexp(log_energy_proposal, axis=1)

    # [batch_size, num_samples]
    tiled_proposal_lse = tf.tile(proposal_lse[tf.newaxis,:], [batch_size, 1])

    # Compute the weights of the observed data.
    # [batch_size, 1]
    log_energy_data = tf.reshape(self.energy_fn(data - self.data_mean), [batch_size])
    tf.summary.histogram("log_energy_data", log_energy_data)
    tf.summary.scalar("min_log_energy_data", tf.reduce_min(log_energy_data))
    tf.summary.scalar("max_log_energy_data", tf.reduce_max(log_energy_data))

    # [batch_size, num_samples]
    tiled_log_energy_data = tf.tile(log_energy_data[:, tf.newaxis], [1, num_samples])

    # Add the weights of the proposal samples with the true data weights.
    # [batch_size, num_samples]
    Z_hat = tf.reduce_logsumexp(
            tf.stack([tiled_log_energy_data, tiled_proposal_lse], axis=-1), axis=-1)
    Z_hat -= tf.log(tf.to_float(self.K+1))
    # Perform the log-sum-exp reduction for IWAE
    # [batch_size]
    Z_hat = tf.reduce_logsumexp(Z_hat, axis=1) - tf.log(tf.to_float(num_samples))

    try:
      # Try giving the proposal lower bound num_samples if it can use it.
      proposal_lp = self.proposal.log_prob(data, num_samples=num_samples)
    except TypeError:
      proposal_lp = self.proposal.log_prob(data)
    lower_bound = proposal_lp + log_energy_data - Z_hat
    return lower_bound

  def sample(self, sample_shape=[1]):
    shape = sample_shape + [self.K]
    proposal_samples = self.proposal.sample(shape) #[sample_shape, K, data_dim]
    log_energy = tf.reshape(
            self.energy_fn(proposal_samples - self.data_mean), shape) #[sample_shape, K]
    indexes = tfd.Categorical(logits=log_energy).sample() #[sample_shape]
    #[sample_shape, data_dim]
    samples = tf.batch_gather(proposal_samples, tf.expand_dims(indexes, axis=-1))
    return tf.squeeze(samples)

class BernoulliNIS(NIS):

  def __init__(self,
               K,
               data_dim,
               energy_hidden_sizes,
               q_hidden_sizes,
               proposal=None,
               data_mean=None,
               temperature=0.7,
               dtype=tf.float32,
               name="nis"):
    self.q_fn = functools.partial(
            base.conditional_bernoulli,
            data_dim=data_dim,
            hidden_sizes=q_hidden_sizes,
            bias_init=data_mean,
            dtype=dtype,
            use_gst=True,
            temperature=temperature,
            name="%s/q" % name)
    super(BernoulliNIS, self).__init__(
            K=K,
            data_dim=data_dim,
            energy_hidden_sizes=energy_hidden_sizes,
            proposal=proposal,
            data_mean=data_mean,
            dtype=dtype,
            name=name)

  def _log_prob(self, data, num_samples=1):
    batch_size = tf.shape(data)[0]
    # Compute log weights for observed data
    # [batch_size]
    log_weights_data = tf.reshape(self.energy_fn(data - self.data_mean), [batch_size])

    # Sample the latent z's from the inference network
    q = self.q_fn(data - self.data_mean)
    # [num_samples, K, batch_size, data_size]
    z_samples = q.sample([num_samples, self.K])

    # [num_samples, K, batch_size, data_size]
    x = z_samples - self.data_mean
    # [num_samples, K, batch_size]
    log_weight_z_samples = tf.reshape(self.energy_fn(x),
            [num_samples, self.K, batch_size])
    # [num_samples, batch_size]
    log_sum_z_weights = tf.reduce_logsumexp(log_weight_z_samples, axis=1)

    # Combine the latent z weight with the observed z weight to form the normalizing
    # constant estimate.
    # [num_samples, batch_size]
    tiled_log_weights_data = tf.tile(log_weights_data[tf.newaxis,:], [num_samples, 1])
    # [num_samples, batch_size]
    Z_hat = tf.reduce_logsumexp(
            tf.stack([tiled_log_weights_data, log_sum_z_weights], axis=-1), axis=-1)
    Z_hat -= tf.log(tf.to_float(self.K+1))

    # Calculate latent log_prob under proposal
    try:
      # Try giving the proposal lower bound extra compute if it can use it.
      # [batch_size]
      proposal_data_lp = self.proposal.log_prob(data, num_samples=num_samples)
      #[num_samples, K, batch_size]
      proposal_z_lp = self.proposal.log_prob(z_samples, num_samples=num_samples)
    except TypeError:
      # [batch_size]
      proposal_lp = self.proposal.log_prob(data)
      #[num_samples, K, batch_size]
      proposal_z_lp = self.proposal.log_prob(z_samples)
    # [num_samples, batch_size]
    proposal_latents_lp_sum = tf.reduce_sum(proposal_z_lp, axis=1)

    # Calculate latent log prob under inference network.
    # [num_samples, K, batch_size]
    q_lp = q.log_prob(z_samples)
    # [num_samples, batch_size]
    q_latents_lp = tf.reduce_sum(q_lp, axis=1)
    # [batch_size]
    expectation_sum = tf.reduce_logsumexp(proposal_latents_lp_sum - q_latents_lp - Z_hat, axis=0)
    lower_bound = proposal_data_lp + log_weights_data + expectation_sum
    return lower_bound

