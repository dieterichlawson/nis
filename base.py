import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions
import functools

def _safe_log(x, eps=1e-8):
  return tf.log(tf.clip_by_value(x, eps, 1.0))

class GSTBernoulli(tfd.Bernoulli):

  def __init__(self,
               temperature,
               logits=None,
               probs=None,
               validate_args=False,
               allow_nan_stats=True,
               name="GSTBernoulli",
               dtype=tf.int32):
    """Construct GSTBernoulli distributions.
    Args:
      temperature: An 0-D `Tensor`, representing the temperature
        of a set of GSTBernoulli distributions. The temperature should be
        positive.
      logits: An N-D `Tensor` representing the log-odds
        of a positive event. Each entry in the `Tensor` parametrizes
        an independent GSTBernoulli distribution where the probability of an
        event is sigmoid(logits). Only one of `logits` or `probs` should be
        passed in.
      probs: An N-D `Tensor` representing the probability of a positive event.
        Each entry in the `Tensor` parameterizes an independent Bernoulli
        distribution. Only one of `logits` or `probs` should be passed in.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.
    Raises:
      ValueError: If both `probs` and `logits` are passed, or if neither.
    """
    with tf.name_scope(name, values=[logits, probs, temperature]) as name:
      self._temperature = tf.convert_to_tensor(
          temperature, name="temperature", dtype=dtype)
      if validate_args:
        with tf.control_dependencies([tf.assert_positive(temperature)]):
          self._temperature = tf.identity(self._temperature)
      super(GSTBernoulli, self).__init__(
              logits=logits,
              probs=probs,
              validate_args=validate_args,
              allow_nan_stats=allow_nan_stats,
              dtype=dtype,
              name=name)

  @property
  def temperature(self):
    """Distribution parameter for the location."""
    return self._temperature

  def _sample_n(self, n, seed=None):
    new_shape = tf.concat([[n], self.batch_shape_tensor()], 0)
    u = tf.random_uniform(new_shape, seed=seed, dtype=self.probs.dtype)
    logistic = _safe_log(u) - _safe_log(1-u)
    hard_sample = tf.cast(tf.greater(self.logits + logistic, 0), self.dtype)
    soft_sample = tf.math.sigmoid((self.logits + logistic)/self.temperature)
    sample = soft_sample + tf.stop_gradient(hard_sample - soft_sample)
    return tf.cast(sample, self.dtype)

  def log_prob(self, value, name="log_prob"):
    lp = super(GSTBernoulli, self).log_prob(value, name=name)
    return tf.reduce_sum(lp, axis=-1)


class MultivariateTruncatedNormal(tfd.TruncatedNormal):

  def log_prob(self, value, name="log_prob"):
    lp = super(MultivariateTruncatedNormal, self).log_prob(value, name=name)
    return tf.reduce_sum(lp, axis=-1)


class MultivariateBernoulli(tfd.Bernoulli):

  def log_prob(self, value, name="log_prob"):
    lp = super(MultivariateBernoulli, self).log_prob(value, name=name)
    return tf.reduce_sum(lp, axis=-1)

def mlp(inputs,
        layer_sizes,
        hidden_activation=tf.math.tanh,
        final_activation=tf.math.log_sigmoid,
        name=None):
  """Creates a simple multi-layer perceptron."""
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    for i, s in enumerate(layer_sizes[:-1]):
      inputs = tf.layers.dense(inputs,
                               units=s,
                               activation=hidden_activation,
                               kernel_initializer=tf.initializers.glorot_uniform,
                               name="layer_%d" % (i+1))
    output = tf.layers.dense(inputs,
                             units=layer_sizes[-1],
                             activation=final_activation,
                             kernel_initializer=tf.initializers.glorot_uniform,
                             name="layer_%d" % (len(layer_sizes)+1))
  return output

def conditional_normal(
        inputs,
        data_dim,
        hidden_sizes,
        hidden_activation=tf.math.tanh,
        scale_min=1e-5,
        truncate=False,
        bias_init=None,
        name=None):
    raw_params = mlp(inputs,
                     hidden_sizes + [2*data_dim],
                     hidden_activation=hidden_activation,
                     final_activation=None,
                     name=name)
    loc, raw_scale = tf.split(raw_params, 2, axis=-1)
    scale = tf.math.maximum(scale_min, tf.math.softplus(raw_scale))
    if bias_init is not None:
      loc = loc + bias_init
    if truncate:
      loc = tf.math.sigmoid(loc)
      return MultivariateTruncatedNormal(loc=loc, scale=scale, low=0., high=1.)
    else:
      return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)

def conditional_bernoulli(
        inputs,
        data_dim,
        hidden_sizes,
        hidden_activation=tf.math.tanh,
        bias_init=None,
        dtype=tf.int32,
        name=None):
    bern_logits = mlp(inputs,
                      hidden_sizes + [data_dim],
                      hidden_activation=hidden_activation,
                      final_activation=None,
                      name=name)
    if bias_init is not None:
      bern_logits = bern_logits + -tf.log(1. / tf.clip_by_value(bias_init, 0.0001, 0.9999) - 1)
    return MultivariateBernoulli(logits=bern_logits, dtype=dtype)

class VAE(object):
  """Variational autoencoder with continuous latent space."""

  def __init__(self, 
               latent_dim,
               data_dim,
               decoder,
               q_hidden_sizes,
               prior=None,
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
    self.decoder = decoder
    self.kl_weight = kl_weight
    with tf.name_scope(name):
      self.q = functools.partial(
            conditional_normal,
            data_dim=latent_dim,
            hidden_sizes=q_hidden_sizes,
            scale_min=scale_min,
            name="q")
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
    batch_size = tf.shape(data)[0]
    data_dim = data.get_shape().as_list()[1]

    # Construct approximate posterior and sample z.
    q_z = self.q(data)  # TODO(dieterichl): Use the train mean to center the data.
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
               scale_min=1e-5,
               dtype=tf.float32,
               truncate=True,
               bias_init=None,
               kl_weight=1.,
               name="gaussian_vae"):
    # Make the decoder with a Gaussian distribution
    with tf.name_scope(name):
      decoder_fn = functools.partial(
            conditional_normal,
            data_dim=data_dim,
            hidden_sizes=decoder_hidden_sizes,
            scale_min=scale_min,
            bias_init=bias_init,
            truncate=truncate,
            name="decoder")

    super().__init__(
            latent_dim=latent_dim, 
            data_dim=data_dim,
            decoder=decoder_fn, 
            q_hidden_sizes=q_hidden_sizes, 
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
               scale_min=1e-5,
               bias_init=None,
               kl_weight=1.,
               dtype=tf.float32,
               name="gaussian_vae"):
    # Make the decoder with a Gaussian distribution
    with tf.name_scope(name):
      decoder_fn = functools.partial(
            conditional_bernoulli,
            data_dim=data_dim,
            hidden_sizes=decoder_hidden_sizes,
            bias_init=bias_init,
            dtype=dtype,
            name="decoder")

    super().__init__(
            latent_dim=latent_dim, 
            data_dim=data_dim, 
            decoder=decoder_fn, 
            q_hidden_sizes=q_hidden_sizes, 
            prior=prior, 
            scale_min=scale_min, 
            kl_weight=kl_weight,
            dtype=dtype,
            name=name)

class NIS(object):

  def __init__(self,
               K, 
               data_dim, 
               energy_hidden_sizes, 
               proposal=None,
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
    self.K = K
    with tf.name_scope(name):
      self.energy_fn = functools.partial(
            mlp,
            layer_sizes=energy_hidden_sizes + [1],
            final_activation=None,
            name="energy_fn_mlp")
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

    # [K, data_size]
    proposal_samples = self.proposal.sample([self.K])
    # [batch_size]
    log_energy_target = tf.reshape(self.energy_fn(data), [batch_size])
    # [K])
    log_energy_proposal = tf.reshape(self.energy_fn(proposal_samples),
            [self.K])
    # [1]
    proposal_lse = tf.reduce_logsumexp(log_energy_proposal, keepdims=True)
    # [batch_size]
    tiled_proposal_lse = tf.tile(proposal_lse, [batch_size])

    # [batch_size]
    denom = tf.reduce_logsumexp(tf.stack([log_energy_target, tiled_proposal_lse], axis=-1), axis=-1)
    denom -= tf.log(tf.to_float(self.K+1))

    try:
      # Try giving the proposal lower bound extra compute if it can use it.
      proposal_lp = self.proposal.log_prob(data, num_samples=num_samples)
    except TypeError:
      proposal_lp = self.proposal.log_prob(data)
    lower_bound = proposal_lp + log_energy_target - denom
    return lower_bound

  def sample(self, sample_shape=[1]):
    shape = sample_shape + [self.K]
    proposal_samples = self.proposal.sample(shape) #[sample_shape, K, data_dim]
    log_energy = tf.reshape(self.energy_fn(proposal_samples), shape) #[sample_shape, K]
    indexes = tfd.Categorical(logits=log_energy).sample() #[sample_shape]
    #[sample_shape, data_dim]
    samples = tf.batch_gather(proposal_samples, tf.expand_dims(indexes, axis=-1)) 
    return samples



