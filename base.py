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
        squash=False,
        bias_init=None,
        name=None):
    raw_params = mlp(inputs,
                     hidden_sizes + [2*data_dim],
                     hidden_activation=hidden_activation,
                     final_activation=None,
                     name=name)
    assert truncate != squash, "Cannot squash and truncate"

    loc, raw_scale = tf.split(raw_params, 2, axis=-1)
    scale = tf.math.maximum(scale_min, tf.math.softplus(raw_scale))
    if bias_init is not None:
      loc = loc + bias_init
    if truncate:
      loc = tf.math.sigmoid(loc)
      return tfd.Independent(
              TruncatedNormal(loc=loc, scale=scale, low=0., high=1.),
              reinterpreted_batch_ndims=1) 
    elif squash:
      return tfd.Independent(
              tfd.TransformedDistribution(
                  distribution=tfd.Normal(loc=loc, scale=scale),
                  bijector=tfp.bijectors.Sigmoid(),
                  name="SigmoidTransformedNormalDistribution"),
              reinterpreted_batch_ndims=1)
    else:
      return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)

def conditional_bernoulli(
        inputs,
        data_dim,
        hidden_sizes,
        hidden_activation=tf.math.tanh,
        bias_init=None,
        dtype=tf.int32,
        reparameterize_gst=False,
        temperature=None,
        name=None):
    bern_logits = mlp(inputs,
                      hidden_sizes + [data_dim],
                      hidden_activation=hidden_activation,
                      final_activation=None,
                      name=name)
    if bias_init is not None:
      bern_logits = bern_logits -tf.log(1. / tf.clip_by_value(bias_init, 0.0001, 0.9999) - 1)

    if reparameterize_gst:
      assert temperature is not None
      base_dist =  GSTBernoulli(temperature, logits=bern_logits, dtype=dtype)
    else:
      base_dist = Bernoulli(logits=bern_logits, dtype=dtype)
    return tfd.Independent(base_dist, reinterpreted_batch_ndims=1)

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
               scale_min=1e-5,
               dtype=tf.float32,
               truncate=True,
               kl_weight=1.,
               name="gaussian_vae"):
    # Make the decoder with a Gaussian distribution
    with tf.name_scope(name):
      decoder_fn = functools.partial(
            conditional_normal,
            data_dim=data_dim,
            hidden_sizes=decoder_hidden_sizes,
            scale_min=scale_min,
            bias_init=data_mean,
            truncate=truncate,
            name="decoder")

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
               name="gaussian_vae"):
    # Make the decoder with a Gaussian distribution
    with tf.name_scope(name):
      decoder_fn = functools.partial(
            conditional_bernoulli,
            data_dim=data_dim,
            hidden_sizes=decoder_hidden_sizes,
            bias_init=data_mean,
            dtype=dtype,
            reparameterize_gst=reparameterize_sample,
            temperature=temperature,
            name="decoder")

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

class NIS(object):

  def __init__(self,
               K,
               data_dim,
               energy_hidden_sizes,
               proposal=None,
               data_mean=None,
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
    if data_mean is not None:
      self.data_mean = data_mean
    else:
      self.data_mean = tf.zeros((), dtype=dtype)
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
    # Sample from the proposal and compute the weighs of the "unseen" samples.
    # We share these across the batch dimension.
    # [num_samples, K, data_size]
    proposal_samples = self.proposal.sample([num_samples, self.K])
    # [num_samples, K]
    log_energy_proposal = tf.reshape(self.energy_fn(proposal_samples - self.data_mean),
            [num_samples, self.K])
    # [num_samples]
    proposal_lse = tf.reduce_logsumexp(log_energy_proposal, axis=1)
    # [batch_size, num_samples]
    tiled_proposal_lse = tf.tile(proposal_lse[tf.newaxis,:], [batch_size, 1])

    # Compute the weights of the observed data.
    # [batch_size, 1]
    log_energy_data = tf.reshape(self.energy_fn(data - self.data_mean), [batch_size])
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
    return samples

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
            conditional_bernoulli,
            data_dim=data_dim,
            hidden_sizes=q_hidden_sizes,
            bias_init=data_mean,
            dtype=dtype,
            reparameterize_gst=True,
            temperature=temperature,
            name="%s/q_mlp" % name)
    super().__init__(
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

def _expand_to_ta(x, length):
  x = tf.convert_to_tensor(x)
  if x.get_shape().ndims == 0:
    expanded = tf.tile(tf.reshape(x, [1]), [length])
  elif x.get_shape().ndims == 1:
    if x.get_shape()[0] == length:
      expanded = x
    else:
      expanded = tf.tile(tf.reshape(x,[1]), [length])
  ta = tf.TensorArray(dtype=x.dtype,
                      size=length,
                      dynamic_size=False,
                      clear_after_read=False,
                      infer_shape=True).unstack(expanded)
  return ta

def _hamiltonian_dynamics(x_0, momentum_0, energy_fn, T, step_size, temps):
  temperature_ta = _expand_to_ta(temps, T)
  t_0 = tf.constant(0, tf.int32, name="t_0")
  x_ta = tf.TensorArray(dtype=x_0.dtype, size=T, dynamic_size=False,
                      clear_after_read=False,infer_shape=True)
  momentum_ta = tf.TensorArray(dtype=momentum_0.dtype, size=T, dynamic_size=False,
                               clear_after_read=False,infer_shape=True)
  kinetic_energy_ta = tf.TensorArray(dtype=x_0.dtype, size=T, dynamic_size=False,
                                     clear_after_read=False,infer_shape=True)
  potential_energy_ta = tf.TensorArray(dtype=x_0.dtype, size=T, dynamic_size=False,
                                       clear_after_read=False,infer_shape=True)
  tas = [x_ta, momentum_ta, kinetic_energy_ta, potential_energy_ta]

  def _step(t, prev_x, prev_momentum, prev_energy_grad, tas):
    temp = temperature_ta.read(t)
    momentum_tilde = prev_momentum - (step_size/2.)*prev_energy_grad
    new_x = prev_x + step_size*momentum_tilde
    energy_at_new_x = tf.squeeze(energy_fn(new_x))
    grad_energy_at_new_x = tf.gradients(energy_at_new_x, new_x)[0]
    new_momentum = temp*(momentum_tilde - (step_size/2.)*grad_energy_at_new_x)
    kinetic_energy = tf.reduce_sum(tf.square(new_momentum)/2.)
    ta_updates = [new_x, new_momentum, kinetic_energy, energy_at_new_x]
    tas = [ta.write(t, z) for ta, z in zip(tas, ta_updates)]
    return (t+1, new_x, new_momentum, grad_energy_at_new_x, tas)

  def _predicate(t, *unused_args):
    return t < T

  grad_energy_0 = tf.gradients(energy_fn(x_0), x_0)[0]
  _, final_x, final_momentum, _, tas = tf.while_loop(
      _predicate,
      _step,
      loop_vars=(t_0, x_0, momentum_0, grad_energy_0, tas))
  xs, momentums, kes, pes = [t.stack() for t in tas]
  return final_x, final_momentum, xs, momentums, kes, pes 


def _reverse_hamiltonian_dynamics(x_T, momentum_T, energy_fn, T, step_size, temps):
  temperature_ta = _expand_to_ta(temps, T)
  t_T = tf.constant(T-1, tf.int32, name="t_T")
  x_ta = tf.TensorArray(dtype=x_T.dtype, size=T, dynamic_size=False,
                      clear_after_read=False,infer_shape=True)
  momentum_ta = tf.TensorArray(dtype=momentum_T.dtype, size=T, dynamic_size=False,
                               clear_after_read=False,infer_shape=True)
  kinetic_energy_ta = tf.TensorArray(dtype=x_T.dtype, size=T, dynamic_size=False,
                                     clear_after_read=False,infer_shape=True)
  potential_energy_ta = tf.TensorArray(dtype=x_T.dtype, size=T, dynamic_size=False,
                                       clear_after_read=False,infer_shape=True)
  tas = [x_ta, momentum_ta, kinetic_energy_ta, potential_energy_ta]

  def _step(t, next_x, next_momentum, next_energy_grad, tas):
    temp = temperature_ta.read(t)
    momentum_tilde = next_momentum/temp + (step_size/2.)*next_energy_grad
    prev_x = next_x - step_size*momentum_tilde
    energy_at_prev_x = tf.squeeze(energy_fn(prev_x))
    grad_energy_at_prev_x = tf.gradients(energy_at_prev_x, prev_x)[0]
    prev_momentum = momentum_tilde + (step_size/2.)*grad_energy_at_prev_x
    kinetic_energy = tf.reduce_sum(tf.square(prev_momentum)/2.)
    ta_updates = [prev_x, prev_momentum, kinetic_energy, energy_at_prev_x]
    new_tas = [ta.write(t, z) for ta, z in zip(tas, ta_updates)]
    return (t-1, prev_x, prev_momentum, grad_energy_at_prev_x, new_tas)

  def _predicate(t, *unused_args):
    return t >= 0

  grad_energy_T = tf.gradients(energy_fn(x_T), x_T)[0]
  _, x_0, momentum_0, _, tas= tf.while_loop(
      _predicate,
      _step,
      loop_vars=(t_T, x_T, momentum_T, grad_energy_T, tas))
  xs, momentums, kes, pes = [t.stack() for t in tas]
  return x_0, momentum_0, xs, momentums, kes, pes

class HIS(object):

  def __init__(self,
               T,
               data_dim,
               energy_hidden_sizes,
               q_hidden_sizes,
               proposal=None,
               data_mean=None,
               init_alpha=1.,
               init_step_size=0.01,
               scale_min=1e-5,
               dtype=tf.float32,
               name="his"):
    self.data_dim = data_dim
    if data_mean is not None:
      self.data_mean = data_mean
    else:
      self.data_mean = tf.zeros((), dtype=dtype)
    self.T = T
    with tf.name_scope(name):
      self.energy_fn = functools.partial(
            mlp,
            layer_sizes=energy_hidden_sizes + [1],
            final_activation=None,
            name="energy_fn_mlp")
      self.q = functools.partial(
            conditional_normal,
            data_dim=data_dim,
            hidden_sizes=q_hidden_sizes,
            scale_min=scale_min,
            bias_init=None,
            truncate=False,
            name="decoder")

      init_alpha = -np.log(1./init_alpha - 1.)
      self.raw_alphas = tf.get_variable(name="raw_alpha",
                                       shape=[T],
                                       dtype=tf.float32,
                                       initializer=tf.constant_initializer(init_alpha),
                                       trainable=False)
      self.alphas = tf.math.sigmoid(self.raw_alphas)
      for i in range(T):
        tf.summary.scalar("alpha_%d" % i, self.alphas[i])
      init_step_size = np.log(np.exp(init_step_size) - 1.)
      self.raw_step_size = tf.get_variable(name="raw_step_size",
                                           shape=[data_dim],
                                           dtype=tf.float32,
                                           initializer=tf.constant_initializer(init_step_size),
                                           trainable=False)
      self.step_size = tf.math.softplus(self.raw_step_size)
      for i in range(data_dim):
        tf.summary.scalar("step_size_%d" % i, self.step_size[i])

    if proposal is None:
      self.proposal = tfd.MultivariateNormalDiag(loc=tf.zeros([2*data_dim], dtype=dtype),
                                                 scale_diag=tf.ones([2*data_dim], dtype=dtype))
    else:
      self.proposal = proposal

  def log_prob(self, data, num_samples=1):
    batch_shape = tf.shape(data)[0:-1]
    reshaped_data = tf.reshape(data, [tf.math.reduce_prod(batch_shape), self.data_dim])
    log_prob = self._log_prob(reshaped_data, num_samples=num_samples)
    log_prob = tf.reshape(log_prob, batch_shape)
    return log_prob

  def _log_prob(self, data, num_samples=1):
    q = self.q(data)
    rho_T = q.sample([num_samples])
    x_T = tf.tile(data[tf.newaxis,:,:], [num_samples, 1,1])
    x_0, rho_0, _, _, _, _ = _reverse_hamiltonian_dynamics(x_T, rho_T, self.energy_fn, self.T,
                                                           step_size=self.step_size, temps=self.alphas)
    log_p0 = self.proposal.log_prob(tf.concat([x_0, rho_0], axis=2))
    elbo = log_p0 - self.data_dim*tf.reduce_sum(tf.log(self.alphas)) - q.log_prob(rho_T)
    return tf.reduce_logsumexp(elbo, axis=0) - tf.log(tf.to_float(num_samples))

  def sample(self, sample_shape=[1]):
    x_and_rho = self.proposal.sample(sample_shape=sample_shape)
    x_0, rho_0 = tf.split(x_and_rho, 2, axis=-1)
    x_T, rho_T, xs, _, _, _ = _hamiltonian_dynamics(x_0, rho_0, self.energy_fn, self.T,
                                                    step_size=self.step_size, temps=self.alphas)
    return x_T, rho_T, xs

