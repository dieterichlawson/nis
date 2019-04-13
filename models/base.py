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
                             name="layer_%d" % len(layer_sizes))
  return output

def conditional_normal(
        inputs,
        data_dim,
        hidden_sizes,
        hidden_activation=tf.math.tanh,
        scale_min=1e-5,
        truncate=False,
        squash=False,
        squash_eps=1e-4,
        bias_init=None,
        scale_init=1.,
        nn_scale=True,
        name=None):
    assert (not truncate) or (not squash), "Cannot squash and truncate"
    if nn_scale:
      raw_params = mlp(inputs,
                       hidden_sizes + [2*data_dim],
                       hidden_activation=hidden_activation,
                       final_activation=None,
                       name=name)
      loc, raw_scale = tf.split(raw_params, 2, axis=-1)
    else:
      loc = mlp(inputs,
                hidden_sizes + [data_dim],
                hidden_activation=hidden_activation,
                final_activation=None,
                name=name + "_loc")
      with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        raw_scale_init = np.log(np.exp(scale_init) - 1 + scale_min)
        raw_scale = tf.get_variable(name="raw_sigma",
                                    shape=[data_dim],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(raw_scale_init),
                                    trainable=True)
    scale = tf.math.maximum(scale_min, tf.math.softplus(raw_scale))
    with tf.name_scope(name):
      tf.summary.histogram("scale", scale, family="scales")
      tf.summary.scalar("min_scale", tf.reduce_min(scale), family="scales")
    if bias_init is not None:
      loc = loc + bias_init
    if truncate:
      loc = tf.math.sigmoid(loc)
      return tfd.Independent(
              TruncatedNormal(loc=loc, scale=scale, low=0., high=1.),
              reinterpreted_batch_ndims=1) 
    elif squash:
      bijectors = [tfp.bijectors.AffineScalar(scale=256.),
                   tfp.bijectors.AffineScalar(shift=-squash_eps/2.,
                                              scale=(1. + squash_eps)),
                   tfp.bijectors.Sigmoid(),
                   ]
      return tfd.Independent(
          tfd.TransformedDistribution(
              distribution=tfd.Normal(loc=loc, scale=scale),
              bijector=tfp.bijectors.Chain(bijectors)
              name="SquashedNormalDistribution"),
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
        use_gst=False,
        temperature=None,
        name=None):
    bern_logits = mlp(inputs,
                      hidden_sizes + [data_dim],
                      hidden_activation=hidden_activation,
                      final_activation=None,
                      name=name)
    if bias_init is not None:
      bern_logits = bern_logits -tf.log(1. / tf.clip_by_value(bias_init, 0.0001, 0.9999) - 1)

    if use_gst:
      assert temperature is not None
      base_dist =  GSTBernoulli(temperature, logits=bern_logits, dtype=dtype)
    else:
      base_dist = tfd.Bernoulli(logits=bern_logits, dtype=dtype)
    return tfd.Independent(base_dist, reinterpreted_batch_ndims=1)
