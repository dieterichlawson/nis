import functools
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from . import base

class NISVIMCO(object):
  """An implementation of NIS with a VAE proposal for discrete data. Uses the VIMCO grad
     estimator."""

  def __init__(self,
               vae,
               K,
               data_dim,
               energy_hidden_sizes,
               data_mean=None,
               dtype=tf.float32,
               name="nis_vimco"):
    """Creates a NIS model with VIMCO gradient estimation.

    Args:
      VAE: The VAE proposal.
      K: The number of proposal samples to take.
      data_dim: The dimension of the data.
      energy_hidden_sizes: The sizes of the hidden layers for the MLP that parameterizes the
        energy function.
    """
    self.dtype=dtype
    self.data_dim = data_dim
    if data_mean is not None:
      self.data_mean = data_mean
    else:
      self.data_mean = tf.zeros((), dtype=dtype)
    self.K = K
    self.proposal = vae
    self.energy_fn = functools.partial(
          base.mlp,
          layer_sizes=energy_hidden_sizes + [1],
          final_activation=None,
          name="%s/energy_fn_mlp" % name)

  def log_prob(self, data, num_samples=1):
    batch_shape = tf.shape(data)[0:-1]
    reshaped_data = tf.reshape(data, [tf.math.reduce_prod(batch_shape), self.data_dim])
    log_prob = self._log_prob(reshaped_data, num_samples=num_samples)
    log_prob = tf.reshape(log_prob, batch_shape)
    return log_prob

  def _log_prob(self, data, num_samples=1):
    batch_size = tf.shape(data)[0]
    # Prior has no params, so stop_grad should be a noop.
    # [K, latent_size]
    vae_z = tf.stop_gradient(self.proposal.prior.sample([self.K]))
    vae_p_x_given_z = self.proposal.decoder(vae_z)
    vae_samples = vae_p_x_given_z.sample([batch_size])  # [B, K, data_dim]

    # [B, K]
    vae_log_p_x_given_z = vae_p_x_given_z.log_prob(vae_samples)
    # stop_grad is probably a noop, as vae_samples is discrete.
    vae_samples = tf.stop_gradient(tf.cast(vae_samples, self.dtype))

    # [B, K]
    log_energy_proposal = tf.reshape(
        tf.squeeze(self.energy_fn(tf.reshape(vae_samples, [-1, self.data_dim]) - self.data_mean), axis=-1),
                   [-1, self.K])
    # [B]
    proposal_lse = tf.reduce_logsumexp(log_energy_proposal, axis=-1)
    #tiled_proposal_lse = tf.tile(proposal_lse[None], [batch_size])

    # [batch_size]
    log_energy_data = tf.squeeze(self.energy_fn(data - self.data_mean), axis=-1)
    Z_hat = tf.reduce_logsumexp(
        tf.stack([log_energy_data, proposal_lse], axis=-1), axis=-1)
    Z_hat -= tf.log(tf.to_float(self.K + 1))

    # Compute LOO baseline
    # [B, K, K]
    loo_baseline = tf.tile(log_energy_proposal[:, None, :], [1, self.K, 1])
    loo_baseline = tf.linalg.set_diag(loo_baseline,
                                      tf.tile(tf.constant(-np.inf, shape=[self.K], dtype=self.dtype)[None, :], [batch_size, 1]))
    # [B, K]
    loo_baseline = (tf.reduce_logsumexp(loo_baseline, axis=-1)
                    + tf.log(tf.cast(self.K, self.dtype))
                    - tf.log(tf.cast(self.K-1, self.dtype)))
    loo_baseline = tf.stack([
        loo_baseline,
        tf.tile(log_energy_data[:, None], [1, self.K])], axis=-1)
    # [B, K]
    loo_baseline = tf.reduce_logsumexp(loo_baseline, axis=-1)
    loo_baseline -= tf.log(tf.to_float(self.K + 1))

    # [B, K]
    learning_signal = tf.stop_gradient(Z_hat[:, None] - loo_baseline)
    tf.summary.histogram("learning_signal", learning_signal)
    tf.summary.scalar("learning_signal_MSE", tf.reduce_mean(tf.square(learning_signal)))

    # [B]
    reinf_grad = tf.reduce_sum(
        learning_signal * vae_log_p_x_given_z,
        axis=-1)
    Z_hat += reinf_grad - tf.stop_gradient(reinf_grad)

    try:
      # Try giving the proposal vae lower bound num_samples if it can use it.
      proposal_lp = self.proposal.log_prob(data, num_samples=num_samples)
    except TypeError:
      proposal_lp = self.proposal.log_prob(data)
    # [B]
    lower_bound = proposal_lp + log_energy_data - Z_hat  # Z_hat_vimco
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
