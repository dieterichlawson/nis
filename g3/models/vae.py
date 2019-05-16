import google3
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import functools

from google3.experimental.users.gjt.his.models import base
import pdb

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

class VAE(object):
  """Variational autoencoder with continuous latent space."""

  def __init__(self,
               latent_dim,
               data_dim,
               decoder,
               q,
               prior=None,
               data_mean=None,
               kl_weight=1.,
               dtype=tf.float32,
               name="vae"):
    """Creates a VAE.

    Args:
      latent_dim: The size of the latent variable of the VAE.
      decoder: A callable that accepts a batch of latent samples and returns a
        distribution over the data space of the VAE. The distribution should
        support sample() and log_prob().
      q: A callable that accepts a batch of data samples and returns a
        distribution over the latent space of the VAE. The distribution should
        support sample() and log_prob().
      prior: A distribution over the latent space of the VAE. The object must
        support sample() and log_prob(). If not provided, defaults to Gaussian.
    """
    self.data_dim = data_dim
    if data_mean is not None:
      self.data_mean = data_mean
    else:
      self.data_mean = tf.zeros((), dtype=dtype)
    self.decoder = decoder
    self.q = q
    self.kl_weight = kl_weight

    self.dtype = dtype
    if prior is None:
      self.prior = tfd.MultivariateNormalDiag(
          loc=tf.zeros([latent_dim], dtype=dtype),
          scale_diag=tf.ones([latent_dim], dtype=dtype))
    else:
      self.prior = prior

  def log_prob(self, data, num_samples=1):
    batch_shape = tf.shape(data)[0:-1]
    reshaped_data = tf.reshape(
        data, [tf.math.reduce_prod(batch_shape), self.data_dim])
    log_prob = self._log_prob(reshaped_data, num_samples=num_samples)
    log_prob = tf.reshape(log_prob, batch_shape)
    return log_prob

  def _log_prob(self, data, num_samples=1):
    mean_centered_data = data - self.data_mean

    # Construct approximate posterior and sample z.
    q_z = self.q(mean_centered_data)
    z = q_z.sample(sample_shape=[num_samples
                                ])  # [num_samples, batch_size, data_dim]
    log_q_z = q_z.log_prob(z)  # [num_samples, batch_size]

    # compute the prior prob of z, #[num_samples, batch_size]
    # Try giving the proposal lower bound extra compute if it can use it.
    try:
      log_p_z = self.prior.log_prob(z, num_samples=num_samples)
    except TypeError:
      log_p_z = self.prior.log_prob(z)

    # Compute the model logprob of the data
    p_x_given_z = self.decoder(z)
    log_p_x_given_z = p_x_given_z.log_prob(data)  # [num_samples, batch_size]

    elbo = (
        tf.reduce_logsumexp(
            log_p_x_given_z + self.kl_weight * (log_p_z - log_q_z), axis=0) -
        tf.log(tf.to_float(num_samples)))
    return elbo

  def sample(self, sample_shape=(1)):
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
    q = functools.partial(
        base.conditional_normal,
        data_dim=latent_dim,
        hidden_sizes=q_hidden_sizes,
        scale_min=scale_min,
        name="%s/q" % name)

    super(GaussianVAE, self).__init__(
        latent_dim=latent_dim,
        data_dim=data_dim,
        decoder=decoder_fn,
        q=q,
        data_mean=data_mean,
        prior=prior,
        kl_weight=kl_weight,
        dtype=dtype,
        name=name)


class ConvGaussianVAE(VAE):
  """ConvVAE with Gaussian generative distribution."""

  def __init__(self,
               latent_dim,
               data_dim,
               base_depth,
               activation=tf.nn.leaky_relu,
               prior=None,
               data_mean=None,
               scale_min=1e-5,
               dtype=tf.float32,
               squash=False,
               kl_weight=1.,
               squash_eps=1e-6,
               name="gaussian_vae"):
    """
        activation: Activation function in hidden layers.
    latent_size: Dimensionality of the encoding.
    output_shape: The output image shape.
    base_depth: Smallest depth for a layer.
    sigma_max: Maximum standard deviation of decoder.
     Make the decoder with a Gaussian distribution
    """
    if data_mean is None:
      data_mean = 0.
    deconv = functools.partial(
        tf.keras.layers.Conv2DTranspose, padding="SAME", activation=activation)
    conv = functools.partial(
        tf.keras.layers.Conv2D, padding="SAME", activation=activation)
    if squash:
      squash_bijector = base.get_squash(squash_eps)
      unsquashed_data_mean = squash_bijector.inverse(data_mean)
      last_decoder_layer = tfpl.DistributionLambda(
          lambda t: tfd.Independent(
              tfd.TransformedDistribution(
                  distribution=tfd.Normal(
                      loc=t[..., :data_dim[-1]] + unsquashed_data_mean,
                      scale=tf.math.maximum(scale_min,
                                            tf.math.softplus(
                                                t[..., data_dim[-1]:]))),
                  bijector=squash_bijector)))
    else:
      last_decoder_layer = tfpl.DistributionLambda(lambda t: tfd.Independent(
          tfd.Normal(
              loc=tf.nn.sigmoid(t[..., :data_dim[-1]]),
              scale=sigma_min + (sigma_max - sigma_min) * tf.nn.sigmoid(t[
                  ..., data_dim[-1]:]))))

    decoder_fn = tf.keras.Sequential([
        tfkl.Lambda(lambda x: x[:, None, None, :]),
        deconv(4 * base_depth, 5, 2),
        deconv(4 * base_depth, 5, 2),
        deconv(2 * base_depth, 5, 2),
        deconv(2 * base_depth, 5, 2),
        deconv(base_depth, 5, 2),
        deconv(base_depth, 5, 2),
        conv(data_dim[-1] * 2, 5, activation=None),
        last_decoder_layer,
    ])
    encoding_layer = tfpl.IndependentNormal
    encoder_fn = tf.keras.Sequential([
        conv(base_depth, 5, 2),
        conv(base_depth, 5, 2),
        conv(2 * base_depth, 5, 2),
        conv(2 * base_depth, 5, 2),
        conv(4 * base_depth, 5, 2),
        conv(4 * latent_dim, 5, 2),
        tfkl.Flatten(),
        tfkl.Dense(encoding_layer.params_size(latent_dim), activation=None),
        encoding_layer(latent_dim),
    ])

    super(ConvGaussianVAE, self).__init__(
        latent_dim=latent_dim,
        data_dim=data_dim,
        decoder=decoder_fn,
        q=encoder_fn,
        data_mean=data_mean,
        prior=prior,
        kl_weight=kl_weight,
        dtype=dtype,
        name=name)

  def log_prob(self, data, num_samples=1):
    return self._log_prob(data, num_samples=num_samples)

  def _log_prob(self, data, num_samples=1):
    mean_centered_data = data - self.data_mean

    # Construct approximate posterior and sample z.
    q_z = self.q(mean_centered_data)
    z = q_z.sample()
    log_q_z = q_z.log_prob(z)  # [num_samples, batch_size]

    # compute the prior prob of z, #[num_samples, batch_size]
    # Try giving the proposal lower bound extra compute if it can use it.
    try:
      log_p_z = self.prior.log_prob(z, num_samples=num_samples)
    except TypeError:
      log_p_z = self.prior.log_prob(z)

    # Compute the model logprob of the data
    p_x_given_z = self.decoder(z)
    log_p_x_given_z = p_x_given_z.log_prob(data)  # [num_samples, batch_size]

    elbo =  log_p_x_given_z + self.kl_weight * (log_p_z - log_q_z)
    return elbo


class BernoulliVAE(VAE):
  """VAE with Bernoulli generative distribution."""

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
    q = functools.partial(
        base.conditional_normal,
        data_dim=latent_dim,
        hidden_sizes=q_hidden_sizes,
        scale_min=scale_min,
        name="%s/q" % name)

    super(BernoulliVAE, self).__init__(
        latent_dim=latent_dim,
        data_dim=data_dim,
        decoder=decoder_fn,
        q=q,
        data_mean=data_mean,
        prior=prior,
        kl_weight=kl_weight,
        dtype=dtype,
        name=name)
