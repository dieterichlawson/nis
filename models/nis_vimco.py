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
    # Sample from the proposal and compute the weights of the "unseen" samples.
    # We share these across the batch dimension.
    # [num_samples, K, latent_size]
    vae_z = tf.stop_gradient(self.proposal.prior.sample([num_samples, self.K]))
    vae_p_x_given_z = self.proposal.decoder(vae_z)
    vae_samples = vae_p_x_given_z.sample()
    # [num_samples, K]
    vae_log_p_x_given_z = vae_p_x_given_z.log_prob(vae_samples)
    vae_samples = tf.stop_gradient(tf.cast(vae_samples, self.dtype))

    # Compute the log energy of the samples from the proposal.
    # [num_samples, K]
    log_energy_proposal = tf.reshape(self.energy_fn(vae_samples - self.data_mean),
            [num_samples, self.K])
    tf.summary.histogram("log_energy_proposal", log_energy_proposal)
    # [num_samples]
    proposal_lse = tf.reduce_logsumexp(log_energy_proposal, axis=1)
    # [batch_size, num_samples]
    tiled_proposal_lse = tf.tile(proposal_lse[tf.newaxis,:], [batch_size, 1])

    # Compute the log energy of the observed data.
    # [batch_size]
    log_energy_data = tf.reshape(self.energy_fn(data - self.data_mean), [batch_size])
    tf.summary.histogram("log_energy_data", log_energy_data)
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
    tf.summary.histogram("Z_hat", Z_hat)

    # Compute part of the learning signal baseline for the gradient estimator.
    # [num_samples, K, K]
    tiled_log_energy_proposal = tf.tile(log_energy_proposal[:,:,tf.newaxis], [1,1, self.K])
    tiled_log_energy_proposal_loo = tf.linalg.set_diag(
            tiled_log_energy_proposal, 
            tf.constant(-np.inf, shape=[num_samples, self.K], dtype=self.dtype))
    # [num_samples, K]
    ls_baseline_loo = tf.reduce_logsumexp(tiled_log_energy_proposal_loo, axis=1)
    # [batch_size, num_samples, K]
    tiled_ls_baseline_loo = tf.tile(ls_baseline_loo[tf.newaxis,:,:], [batch_size, 1, 1])

    # [batch_size, num_samples, K]
    loo_baseline = tf.reduce_logsumexp(
            tf.stack([tiled_ls_baseline_loo, 
                     tf.tile(tiled_log_energy_data[:,:,tf.newaxis], [1, 1, self.K])],
                     axis=-1),
            axis=-1)
    # Remove the num_samples dimension and divide by K-1
    # [batch_size, K]
    loo_baseline = tf.reduce_logsumexp(loo_baseline, axis=1) 
    loo_baseline -= tf.log(tf.to_float(self.K))
    # [batch_size, K]
    learning_signal = tf.stop_gradient(Z_hat[:,tf.newaxis] - loo_baseline)
    tf.summary.histogram("learning_signal", learning_signal)
    tf.summary.histogram("learning_signal_abs", tf.math.abs(learning_signal))
    tf.summary.scalar("avg_learning_signal_abs", tf.reduce_mean(tf.math.abs(learning_signal)))
    
    score_fn = learning_signal*tf.tile(
            tf.reduce_sum(vae_log_p_x_given_z, axis=0)[tf.newaxis,:], [batch_size, 1])
    Z_hat_vimco = Z_hat + tf.reduce_sum(score_fn - tf.stop_gradient(score_fn), axis=-1)
 
    try:
      # Try giving the proposal vae lower bound num_samples if it can use it.
      proposal_lp = self.proposal.log_prob(data, num_samples=num_samples)
    except TypeError:
      proposal_lp = self.proposal.log_prob(data)
    lower_bound = proposal_lp + log_energy_data - Z_hat_vimco
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
