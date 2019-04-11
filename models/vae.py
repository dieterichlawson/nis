import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions
import functools

from . import base

class VAE(object):
  """Variational autoencoder with continuous latent space."""

  def __init__(self,
               latent_dim,
               data_dim,
               decoder,
               q_hidden_sizes,
               prior=None,
               data_mean=None,
               scale_min=1e-5,
               kl_weight=1.,
               dtype=tf.float32,
               name="vae"):
    """Creates a VAE.

    Args:
      latent_dim: The size of the latent variable of the VAE.
      decoder: A callable that accepts a batch of latent samples and returns a distribution
        over the data space of the VAE. The distribution should support sample() and
        log_prob().
      q_hidden_sizes: A list of python ints, the sizes of the hidden layers in the MLP
        that parameterizes the q of this VAE.
      prior: A distribution over the clatent space of the VAE. The object must support
        sample() and log_prob(). If not provided, defaults to Gaussian.
    """
    self.data_dim = data_dim
    if data_mean is not None:
      self.data_mean = data_mean
    else:
      self.data_mean = tf.zeros((), dtype=dtype)
    self.decoder = decoder
    self.kl_weight = kl_weight
    self.q = functools.partial(
          base.conditional_normal,
          data_dim=latent_dim,
          hidden_sizes=q_hidden_sizes,
          scale_min=scale_min,
          name="%s/q" % name)
    self.dtype = dtype
    if prior is None:
      self.prior = tfd.MultivariateNormalDiag(loc=tf.zeros([latent_dim], dtype=dtype),
                                              scale_diag=tf.ones([latent_dim], dtype=dtype))
    else:
      self.prior = prior

  def log_prob(self, data, num_samples=1):
    batch_shape = tf.shape(data)[0:-1]
    reshaped_data = tf.reshape(data, [tf.math.reduce_prod(batch_shape), self.data_dim])
    log_prob = self._log_prob(reshaped_data, num_samples=num_samples)
    log_prob = tf.reshape(log_prob, batch_shape)
    return log_prob

  def _log_prob(self, data, num_samples=1):
    mean_centered_data = data - self.data_mean
    batch_size = tf.shape(data)[0]
    data_dim = data.get_shape().as_list()[1]

    # Construct approximate posterior and sample z.
    q_z = self.q(mean_centered_data)
    z = q_z.sample(sample_shape=[num_samples]) #[num_samples, batch_size, data_dim]
    log_q_z = q_z.log_prob(z) #[num_samples, batch_size]

    # compute the prior prob of z, #[num_samples, batch_size]
    # Try giving the proposal lower bound extra compute if it can use it.
    try:
      log_p_z = self.prior.log_prob(z, num_samples=num_samples)
    except TypeError:
      log_p_z = self.prior.log_prob(z)

    # Compute the model logprob of the data
    p_x_given_z = self.decoder(z)
    log_p_x_given_z = p_x_given_z.log_prob(data) #[num_samples, batch_size]

    elbo = (tf.reduce_logsumexp(log_p_x_given_z + self.kl_weight*(log_p_z - log_q_z), axis=0) -
            tf.log(tf.to_float(num_samples)))
    return elbo

  def sample(self, sample_shape=[1]):
    z = self.prior.sample(sample_shape)
    p_x_given_z = self.decoder(z)
    return tf.cast(p_x_given_z.sample(), self.dtype)

class GaussianVAE(VAE):
  """VAE with Gaussian generative distribution."""

  def __init__(self,
               latent_dim,
               data_dim,
               decoder_hidden_sizes,
               q_hidden_sizes,
               prior=None,
               data_mean=None,
               decoder_nn_scale=True,
               scale_min=1e-5,
               dtype=tf.float32,
               squash=False,
               kl_weight=1.,
               name="gaussian_vae"):
    # Make the decoder with a Gaussian distribution
    decoder_fn = functools.partial(
            base.conditional_normal,
            data_dim=data_dim,
            hidden_sizes=decoder_hidden_sizes,
            scale_min=scale_min,
            nn_scale=decoder_nn_scale,
            bias_init=data_mean,
            truncate=False,
            squash=squash,
            name="%s/decoder" % name)

    super().__init__(
            latent_dim=latent_dim,
            data_dim=data_dim,
            decoder=decoder_fn,
            q_hidden_sizes=q_hidden_sizes,
            data_mean=data_mean,
            prior=prior,
            scale_min=scale_min,
            kl_weight=kl_weight,
            dtype=dtype,
            name=name)

class BernoulliVAE(VAE):
  """VAE with Gaussian generative distribution."""

  def __init__(self,
               latent_dim,
               data_dim,
               decoder_hidden_sizes,
               q_hidden_sizes,
               prior=None,
               data_mean=None,
               scale_min=1e-5,
               kl_weight=1.,
               reparameterize_sample=False,
               temperature=None,
               dtype=tf.float32,
               name="bernoulli_vae"):
    # Make the decoder with a Gaussian distribution
    decoder_fn = functools.partial(
          base.conditional_bernoulli,
          data_dim=data_dim,
          hidden_sizes=decoder_hidden_sizes,
          bias_init=data_mean,
          dtype=dtype,
          use_gst=reparameterize_sample,
          temperature=temperature,
          name="%s/decoder" % name)

    super().__init__(
            latent_dim=latent_dim,
            data_dim=data_dim,
            decoder=decoder_fn,
            q_hidden_sizes=q_hidden_sizes,
            data_mean=data_mean,
            prior=prior,
            scale_min=scale_min,
            kl_weight=kl_weight,
            dtype=dtype,
            name=name)
